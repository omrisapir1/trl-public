import json
import random
import re
import time

from vllm import SamplingParams
import numpy as np

MAX_THINK_TOKENS = 256
MAX_END_TOKENS = 512
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
REPETITION_PENALTY = 1.0
MODEL = 'omrisap/Qwen2.5-1.5B_30K_COT_SFT'
THINK_END_TOKEN = '</think>'
THINK_START_TOKEN = '<think>'
THINK_BOTH_TOKEN = '<think></think>'

ANSWER_END_TOKEN = '</answer>'
ANSWER_START_TOKEN = '<answer>'
END_OF_TEXT_ID_TOKEN = 151643

UNIFIED_MAX_TOKENS = 512
MAX_FIRST_ANS_TOKENS = 2048

N_TOTAL_SPLITS = 4
LAST_SPLIT = 4
CORRECT_STRUCTURE_REWARD = 0.1
CORRECT_FLOAT_REWARD = 0.1

class TreeOfThoughts:
    def __init__(self, llm, max_split_depth=34, max_depth=9):
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
            stop=["</think>", '</answer>','<answer>'],
            n=1  # Generate one continuation per prompt
        )
        self.last_sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=UNIFIED_MAX_TOKENS,#MAX_END_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            stop=['</answer>'],
            n=1  # Generate one continuation per prompt
        )
        # self.no_more_splits_sampling_params = SamplingParams(
        #     temperature=TEMPERATURE,
        #     max_tokens=UNIFIED_MAX_TOKENS,
        #     # (self.max_depth - self.max_split_depth) * MAX_THINK_TOKENS + MAX_END_TOKENS,
        #     top_p=TOP_P,
        #     top_k=TOP_K,
        #     repetition_penalty=REPETITION_PENALTY,
        #     skip_special_tokens=False,
        #     stop=['</answer>'],
        #     n=1  # Generate one continuation per prompt
        # )
        self.first_full_ans = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_FIRST_ANS_TOKENS,
            # (self.max_depth - self.max_split_depth) * MAX_THINK_TOKENS + MAX_END_TOKENS,
            # top_p=TOP_P,
            # top_k=TOP_K,
            # repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            stop=['</answer>','<answer>'],
            n=1  # Generate one continuation per prompt
        )

    def is_correct_solution(self,answer, numerical_label):
        """Extract the last numerical answer enclosed in \boxed{}"""
        total_reward = 0
        matches = re.findall(r'\\boxed\{([^}]*)\}', answer)
        if matches:
            extracted_answer = re.sub(r'[^0-9.-]', '', matches[-1]).strip()
            if extracted_answer.replace('.', '', 1).replace('-','',1).isdigit():

                try:
                    extracted_answer = float(extracted_answer)
                    total_reward += CORRECT_FLOAT_REWARD
                    total_reward += int(float(extracted_answer) == float(numerical_label))
                    return total_reward
                except ValueError:
                    return total_reward
        return total_reward

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
        elif node.get('next_split'):
            return node['next_split']
        if node['depth'] == 0:
            return 2
        elif node['depth'] == 1:
            return 2
        elif node['depth'] == 2:
            return 3
        return 1


    def generate_tree(self, problem, numerical_label):
        """Build the tree structure with parent-child relationships"""
        # Tree structure: list of dictionaries with 'text' and 'parent_idx'
        problem_as_chat_prompt = self.preprocess_problem(problem)
        tree = [{
            'text': problem_as_chat_prompt + THINK_BOTH_TOKEN,
            'prompt': '',
            'parent_idx': None,
            'depth': 0,
            'split': -1,
            'last_chance':False
        }]
        # first_full_output = self.llm.generate([tree[0]['text']], self.first_full_ans)
        # first_full_completion = first_full_output[0].outputs[0]
        # if first_full_completion.finish_reason == 'length' or first_full_completion.stop_reason == END_OF_TEXT_ID_TOKEN or first_full_completion.stop_reason is None:
        #     print('SKIPPED')
        #     return tree, []
        # full_ans = first_full_completion.text
        # thoughts_count = full_ans.count(THINK_END_TOKEN) + 1
        #
        # if thoughts_count > N_TOTAL_SPLITS:
        #     start_index = kth_occurrence_from_end(full_ans, THINK_BOTH_TOKEN, N_TOTAL_SPLITS) + len(THINK_BOTH_TOKEN)
        #     tree[0]['text'] += full_ans[:start_index]



        # completions = [output.outputs[0] for output in first_full_output]
        # all_prompts_token_ids = [output.prompt_token_ids for output in outputs]

        current_depth = 0
        final_nodes = []
        logs = []
        counter_max_depth, counter_not_max_depth = 0, 0
        while current_depth <= self.max_depth:
            # Get unsplit nodes at current depth
            candidates, idxs = zip(*[(n, i) for i, n in enumerate(tree) if n['depth'] == current_depth])

            # Collect nodes to split and their split counts
            split_nodes = []
            for node, idx in zip(candidates, idxs):


                if current_depth == self.max_depth:
                    counter_max_depth += 1
                    logs.append(f'ADDed in MAX depth {counter_max_depth} and idx is {idx}')
                    final_nodes.append((node['parent_idx'], 0))
                    continue
                split_count = self.decide_split(node)
                node['split'] = split_count
                if split_count:
                    split_nodes.append((node, split_count, idx))
                else:
                    logs.append(f'ADDed not in max depth {counter_not_max_depth} and idx is {idx}')
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
            if current_depth == (self.max_depth-1):
                outputs = self.llm.generate(batch_prompts, self.last_sampling_params)
            else:
                outputs = self.llm.generate(batch_prompts, self.sampling_params)
            completions = [output.outputs[0] for output in outputs]
            all_prompts_token_ids = [output.prompt_token_ids for output in outputs]

            # Assign children to parents
            comp_idx = 0
            for parent_idx, split_count in split_mapping:
                parent = tree[parent_idx]

                # Add children
                children_completions = completions[comp_idx:comp_idx + split_count]
                prompts_token_ids = all_prompts_token_ids[comp_idx:comp_idx + split_count]
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
                        stop_token = children_completion.stop_reason
                        if stop_token and not parent.get('to_stop'):
                            print(stop_token)
                            raise
                        text = ' ' + text
                    else:
                        parent['last_chance'] = False
                        stop_token = children_completion.stop_reason
                        text += stop_token
                        if parent.get('predict_answer') and any(t in text for t in [THINK_END_TOKEN, THINK_START_TOKEN, ANSWER_START_TOKEN]):
                            parent['reward'] = 0
                            parent['to_stop'] = True
                        elif stop_token == ANSWER_END_TOKEN:
                            parent['reward'] = self.is_correct_solution(text, numerical_label) + CORRECT_STRUCTURE_REWARD
                            parent['to_stop'] = True

                        elif stop_token == ANSWER_START_TOKEN:
                            parent['next_split'] = LAST_SPLIT
                            parent['predict_answer'] = True


                    parent['completion_ids'] += children_completion.token_ids


                    parent['text'] += text
                    parent['depth'] += 1
                    continue


                for children_completion, prompt_token_ids in zip(children_completions, prompts_token_ids):

                    node = {
                        'prompt': parent['prompt'] + parent['text'],
                        'parent_idx': parent_idx,
                        'depth': current_depth + 1,
                        'completion_ids': children_completion.token_ids,
                        'prompt_ids': prompt_token_ids,
                    }
                    text = children_completion.text
                    if children_completion.finish_reason == 'length':
                        node['last_chance'] = True
                        node['predict_answer'] = parent.get('predict_answer')
                        text = ' ' + text
                    elif children_completion.stop_reason == THINK_END_TOKEN:
                        text += THINK_END_TOKEN
                    elif children_completion.stop_reason == ANSWER_END_TOKEN:
                        text += ANSWER_END_TOKEN
                        node['reward'] = self.is_correct_solution(text, numerical_label) + CORRECT_STRUCTURE_REWARD
                        node['to_stop'] = True
                    elif children_completion.stop_reason == END_OF_TEXT_ID_TOKEN or children_completion.stop_reason is None:
                        node['reward'] = 0
                        node['to_stop'] = True
                        node['predict_answer'] = parent.get('predict_answer')
                    elif children_completion.stop_reason == ANSWER_START_TOKEN:
                        text += ANSWER_START_TOKEN
                        node['next_split'] = LAST_SPLIT
                        node['predict_answer'] = True
                    if parent.get('predict_answer') and any(t in text for t in [THINK_END_TOKEN, THINK_START_TOKEN, ANSWER_START_TOKEN]):
                        node['reward'] = 0
                        node['to_stop'] = True
                    node['text'] = text
                    tree.append(node)

            current_depth += 1

        return tree, final_nodes, logs

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
        try:
            assert sorted([n['reward'] for n in tree if n.get('reward',-111) >=0 and not n.get('rewards')]) == sorted(tree[0]['rewards'])
        except:
            print(logs)
            print(tree)
            raise

        for n in tree:
            rewards = n.get('rewards',[1])
            std = np.std(rewards)
            if std:
                n['reward'] = np.mean(rewards)
            else:
                n['dont_calc_loss'] = True

        if random.random() < 1:
            json.dump(tree,open(f'/workspace/Data_in_training/{time.time()}','w'))
        else:
            json.dump([{'rewards':t.get('rewards'), 'reward':t.get('reward'), 'parent_idx':t.get('parent_idx')} for t in tree], open(f'/workspace/Data_in_training/{time.time()}', 'w'))


def kth_occurrence_from_end(s, sub, k):
    """
    Returns the index of the k'th occurrence of `sub` in `s` from the end.
    If the substring doesn't occur k times, returns -1.
    """
    pos = len(s)
    for i in range(k):
        pos = s.rfind(sub, 0, pos)
        if pos == -1:
            return -1
    return pos