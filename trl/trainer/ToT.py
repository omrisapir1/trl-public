
import re

from vllm import SamplingParams
import numpy as np

MAX_THINK_TOKENS = 256
MAX_END_TOKENS = 512
TEMPERATURE = 1.3#0.9
TOP_P = 0.95#1.0
TOP_K = 100#50
REPETITION_PENALTY = 1.0
MODEL = 'omrisap/Qwen2.5-1.5B_30K_COT_SFT'
THINK_TAG_START = '<think></think>'
THINK_END_TOKEN = '</think>'
ANSWER_END_TOKEN = '</answer>'
END_OF_TEXT_ID_TOKEN = 151643

class TreeOfThoughts:
    def __init__(self, llm, max_split_depth=3, max_depth=25):
        self.llm = llm
        self.max_depth = max_depth
        self.max_split_depth = max_split_depth
        self.tokenizer= self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_THINK_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            stop=["</think>", '</answer>'],
            n=1  # Generate one continuation per prompt
        )
        self.last_sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=MAX_END_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            stop=['</answer>'],
            n=1  # Generate one continuation per prompt
        )
        self.no_more_splits_sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=(self.max_depth - self.max_split_depth) * MAX_THINK_TOKENS + MAX_END_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            stop=['</answer>'],
            n=1  # Generate one continuation per prompt
        )

    def is_correct_solution(self,answer, numerical_label):
        """Extract the last numerical answer enclosed in \boxed{}"""
        matches = re.findall(r'\\boxed\{([^}]*)\}', answer)
        if matches:
            extracted_answer = re.sub(r'[^0-9.-]', '', matches[-1]).strip()
            if extracted_answer.replace('.', '', 1).replace('-','',1).isdigit():
                try:
                    return int(float(extracted_answer) == float(numerical_label))
                except ValueError:
                    return 0
        return 0

    def preprocess_problem(self, problem):
        # Ensure the contents are strings

        # Get the formatted conversation text (without tokenization)
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": problem},
                # {"role": "assistant", "content": solution}
            ],
            tokenize=False,  # Do not tokenize here
            add_generation_prompt=True,  # No extra prompt tokens added
            continue_final_message=False
        )
        # Now tokenize the resulting text (ensuring truncation if needed)
        return text

    def decide_split(self, node):
        if node.get('to_stop'):
            return 0
        elif node.get('last_chance'):
            return 1
        if node['depth'] == 0:
            return 4
        elif node['depth'] == 1:
            return 3
        elif node['depth'] == 2:
            return 2
        elif node['depth'] == 3:
            return 2
        return 1


    def generate_tree(self, problem, numerical_label):
        """Build the tree structure with parent-child relationships"""
        # Tree structure: list of dictionaries with 'text' and 'parent_idx'
        problem_as_chat_prompt = self.preprocess_problem(problem)
        tree = [{
            'text': problem_as_chat_prompt + THINK_TAG_START,
            'prompt': '',
            'parent_idx': None,
            'depth': 0,
            'split': None,
            'last_chance':False
        }]

        current_depth = 0
        final_nodes = []
        while current_depth <= self.max_depth:
            # Get unsplit nodes at current depth
            candidates, idxs = zip(*[(n, i) for i, n in enumerate(tree) if n['depth'] == current_depth])

            # Collect nodes to split and their split counts
            split_nodes = []
            for node, idx in zip(candidates, idxs):
                if current_depth == self.max_depth:
                    final_nodes.append((node['parent_idx'], 0))
                    continue
                split_count = self.decide_split(node)
                node['split'] = split_count
                if split_count:
                    split_nodes.append((node, split_count, idx))
                else:
                    final_nodes.append((node['parent_idx'], node['reward']))

            if not split_nodes:
                break

            # Prepare batch prompts: [node1_prompt * split_count1, node2_prompt * split_count2, ...]
            batch_prompts = []
            split_mapping = []  # Tracks (tree_idx, split_count)
            for node, split_count, idx in split_nodes:
                batch_prompts.extend([node.get('prompt','') + node.get('text','')] * split_count)
                split_mapping.append((idx, split_count))

            # Generate all continuations in single batch
            if current_depth == self.max_split_depth:
                outputs = self.llm.generate(batch_prompts, self.no_more_splits_sampling_params)
            elif current_depth == (self.max_depth-1):
                outputs = self.llm.generate(batch_prompts, self.last_sampling_params)
            else:
                outputs = self.llm.generate(batch_prompts, self.sampling_params)
            completions = [output.outputs[0] for output in outputs]
            prompts_token_ids = [output.prompt_token_ids for output in outputs]
            # completions = [f'random_text {current_depth}_{i}' for i in range(len(batch_prompts))]

            # Assign children to parents
            comp_idx = 0
            for parent_idx, split_count in split_mapping:
                parent = tree[parent_idx]

                # Add children
                children_completions = completions[comp_idx:comp_idx + split_count]
                comp_idx += split_count
                if split_count == 1:
                    children_completion = children_completions[0]
                    text = children_completion.text
                    if children_completion.finish_reason == 'length' or children_completion.stop_reason == END_OF_TEXT_ID_TOKEN or children_completion.stop_reason is None:
                        if parent.get('last_chance') or children_completion.stop_reason == END_OF_TEXT_ID_TOKEN or children_completion.stop_reason is None:
                            parent['to_stop'] = True
                            parent['reward'] = 0
                        else:
                            parent['last_chance'] = True
                        text = ' ' + text
                    else:

                        stop_token = children_completion.stop_reason
                        text += stop_token

                        if stop_token == ANSWER_END_TOKEN:
                            parent['reward'] = self.is_correct_solution(text, numerical_label)
                            parent['to_stop'] = True
                    parent['last_chance'] = False

                    parent['completion_ids'] += children_completion.token_ids


                    parent['text'] += text
                    parent['prompt'] += text
                    parent['depth'] += 1
                    continue


                for children_completion, prompt_token_ids in zip(children_completions, prompts_token_ids):
                    node = {
                        'orig_prompt':parent['prompt'] + parent['text'],
                        'prompt': parent['prompt'] + parent['text'],
                        'parent_idx': parent_idx,
                        'depth': current_depth + 1,
                        'completion_ids': children_completion.token_ids,
                        'prompt_ids': prompt_token_ids,
                    }
                    text = children_completion.text
                    if children_completion.finish_reason == 'length':
                        node['last_chance'] = True
                        text = ' ' + text
                    elif children_completion.stop_reason == THINK_END_TOKEN:
                        text += THINK_END_TOKEN
                    elif children_completion.stop_reason == ANSWER_END_TOKEN:
                        text += ANSWER_END_TOKEN
                        node['reward'] = self.is_correct_solution(text, numerical_label)
                        node['to_stop'] = True
                    elif children_completion.stop_reason == END_OF_TEXT_ID_TOKEN:
                        node['reward'] = 0
                        node['to_stop'] = True
                    node['text'] = text
                    tree.append(node)

            current_depth += 1

        return tree, final_nodes

    def propogate_reward(self, node_idx, reward, tree):
        node = tree[node_idx]
        if not isinstance(node.get('rewards'), list):
            node['rewards'] = [reward]
        else:
            node['rewards'].append(reward)
        if node['parent_idx'] is not None:
            return self.propogate_reward(node['parent_idx'], reward, tree)



    def evaluate_tree(self, tree, final_nodes):
        for node in final_nodes:
            self.propogate_reward(node[0], node[1], tree)
        for n in tree:
            rewards = n.get('rewards',[1])
            std = np.std(rewards)
            if std:
                n['reward'] = np.mean(rewards)
            else:
                n['dont_calc_loss'] = True

