import json
import os.path
import random
import time
import re
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from vllm import SamplingParams, LLM
from .extract_answer import extract_final_answer, math_equal


TEMPARTURE = 1.3
TOP_K = 100
REPETITION_PENALTY = 1.1
TOP_P = 0.95

class NodeState(Enum):
    EXPLORING = 1
    TERMINAL = 2
    INVALID = 3

class StopReason(Enum):
    LENGTH = 1
    END_TOKEN = 2
    FIND_BOX_ANSWER = 3

PATH_TO_SAVE_DATA = 'training_data/'

class TreeNode:
    def __init__(self, prompt_text: str, completion_text: str = "", parent: Optional['TreeNode'] = None):
        self.prompt_text = prompt_text         # The original prompt/context
        self.completion_text = completion_text # The generated output
        self.parent = parent
        self.children: List[TreeNode] = []
        self.state = NodeState.EXPLORING
        self.reward: Optional[float] = None
        self.rewards: List[float] = []
        self.depth = 0 if parent is None else parent.depth + 1
        self.prompt_ids: List[int] = []
        self.completion_ids: List[int] = []
        self.next_split: Optional[int] = None
        self.stop_reason: Optional[StopReason] = None
        self.truncated: bool = False
        self.structured_reward: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_text": self.prompt_text,
            "completion_text": self.completion_text,
            "state": self.state.name,
            "reward": self.reward,
            "rewards": self.rewards,
            "depth": self.depth,
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "next_split": self.next_split,
            "stop_reason": self.stop_reason.name if self.stop_reason else None,
            "children": [child.to_dict() for child in self.children],
            "truncated": self.truncated,
            "structured_reward": self.structured_reward,
        }

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

    def mark_terminal(self, reward: float, stop_reason: StopReason):
        self.state = NodeState.TERMINAL
        self.reward = reward
        self.stop_reason = stop_reason

    def is_terminal(self) -> bool:
        return self.state in {NodeState.TERMINAL, NodeState.INVALID}

    def propagate_reward(self, reward: float):
        self.rewards.append(reward)
        if self.parent:
            self.parent.propagate_reward(reward)

    def compute_final_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0


# ---------------------- TreeOfThoughts Class ---------------------------
class TreeOfThoughts:
    # Configuration constants
    THINK_END_TOKEN = '</think>'
    THINK_START_TOKEN = '<think>'
    END_OF_TEXT_ID_TOKEN = 151643

    MAX_THINK_TOKENS = 512
    MAX_MID_TO_FINAL_TOKENS = 1700
    MAX_FIRST_ANS_TOKENS = 2200 + 256
    MAX_INVALID_TOKENS_TO_CALC_LOSS_FOR = 3500

    CORRECT_STRUCTURE_REWARD = 0.01
    FIRST_SPLIT_COUNT = 2
    FIRST_SPLIT_PROB = 1
    MIN_THINK_TAG_SPLIT = 1
    LAST_SPLIT_COUNT = 2
    LAST_SPLIT_PROB = 1
    MID_SPLIT_COUNT = 2
    MID_SPLIT_PROB = 1

    NON_SPLIT_COUNT = 1
    SPLIT_LEVELS = [9, 12, 14]
    SPLIT_COUNTES = [4, 5, 6]


    def __init__(self, llm):
        self.llm = llm
        self.tokenizer = self.llm.get_tokenizer()

        # Define sampling parameters for different generation stages.
        self.think_sampling_params = SamplingParams(
            temperature=TEMPARTURE,
            max_tokens=self.MAX_THINK_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            stop=[self.THINK_END_TOKEN],
            n=1,
            include_stop_str_in_output=True,
        )
        self.mid_to_end_sampling_params = SamplingParams(
            temperature=TEMPARTURE,
            max_tokens=self.MAX_MID_TO_FINAL_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            n=1,
            include_stop_str_in_output=True,
        )
        self.final_sampling_params = SamplingParams(
            temperature=TEMPARTURE,
            max_tokens=self.MAX_THINK_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            n=1,
            include_stop_str_in_output=True,
        )
        self.first_answer_params = SamplingParams(
            temperature=TEMPARTURE,
            max_tokens=self.MAX_FIRST_ANS_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            skip_special_tokens=False,
            n=1,
            include_stop_str_in_output=True,
        )
        self._remove_logs()

    def _remove_logs(self):
        if not os.path.exists(PATH_TO_SAVE_DATA):
            os.makedirs(PATH_TO_SAVE_DATA)
        for cur_f in os.listdir(PATH_TO_SAVE_DATA):
            try:
                os.remove(os.path.join(PATH_TO_SAVE_DATA, cur_f))
            except:
                pass

    def preprocess_problem(self, problem: str) -> str:
        """
        Process the problem prompt using the LLM's chat template.
        """
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
            continue_final_message=False
        )
        # For the root, we combine the prompt with an initial thinking context.
        return prompt

    def decide_split(self, node: TreeNode) -> int:
        """
        Decide the number of child branches to generate.
        """
        if node.is_terminal():
            return 0
        return self.MID_SPLIT_COUNT
        # elif node.depth < (self.max_split_depth-1):
        #     return self.MID_SPLIT_COUNT
        # else:
        #     return self.LAST_SPLIT_COUNT


    def get_all_nodes(self, node: 'TreeNode') -> List[TreeNode]:
        nodes = [node]
        for n in node.children:
            nodes += self.get_all_nodes(n)
        return nodes

    def handle_stop_conditions(self, completion: Any, text: str, final_answer: Optional[str] = None, initial: bool = False) -> Dict[str, Optional[Any]]:
        """
        Centralized handling of stop conditions based on LLM output.

        Returns a dictionary with keys:
          - "to_stop": bool, whether the node should be terminal.
          - "reward": Optional[float], reward to assign.
          - "next_split": Optional[int], if further splitting is required.
          - "text": The (possibly modified) text.
          - "stop_reason": Optional[StopReason]
        """
        result = {"to_stop": False, "reward": None, "next_split": None, "text": text, "stop_reason": None}


        if initial and self.THINK_START_TOKEN not in text:
            result["to_stop"] = True
            result["reward"] = self.evaluate_solution(text, final_answer)
            result["stop_reason"] = StopReason.LENGTH if completion.finish_reason == 'length' else StopReason.END_TOKEN
            return result
        elif initial:
            return result

        if completion.finish_reason == 'length' or completion.stop_reason == self.END_OF_TEXT_ID_TOKEN or completion.stop_reason is None:
            result["to_stop"] = True
            result["reward"] = result["reward"] = self.evaluate_solution(text, final_answer)
            result["stop_reason"] = StopReason.LENGTH if completion.finish_reason == 'length' else StopReason.END_TOKEN
            return result
        elif re.search(r"\\boxed\s*\{(.*?)\}", text, re.DOTALL | re.IGNORECASE):
            result["to_stop"] = True
            result["reward"] = result["reward"] = self.evaluate_solution(text, final_answer)
            result["stop_reason"] = StopReason.FIND_BOX_ANSWER
            return result

        return result

    def evaluate_solution(self, text: str, final_answer: str) -> float:
        """
        Evaluate the solution by extracting a numerical answer from a \boxed{...} pattern.
        """
        final_prediction = extract_final_answer(text)
        return int(math_equal(final_prediction, final_answer))


    def evaluate_sturcture_reward(self, text: str) -> float:
        tags = [m.group() for m in re.finditer(r'</?think>', text)]
        if len(tags) > 0 and not text.startswith(self.THINK_START_TOKEN):
            return 0
        stack = []

        for tag in tags:
            if tag == self.THINK_START_TOKEN:
                if stack:
                    return 0
                stack.append(tag)
            elif tag == self.THINK_END_TOKEN:
                if not stack:
                    return 0
                stack.pop()
        return (not stack) * self.CORRECT_STRUCTURE_REWARD


    def initial_generation(self, root: TreeNode) -> List[TreeNode]:
        """
        Generate the first set of child nodes from the root.
        """
        full_prompt = root.prompt_text
        prompts = [full_prompt] * (self.FIRST_SPLIT_COUNT if random.random() < self.FIRST_SPLIT_PROB else self.NON_SPLIT_COUNT)
        outputs = self.llm.generate(prompts, self.first_answer_params)
        first_level_nodes = []
        valid_branch_found = False

        for output in outputs:
            completion = output.outputs[0]
            full_text = completion.text
            stop_info = self.handle_stop_conditions(completion, full_text, initial=True)

            # Create a node with separated prompt and completion.
            node = TreeNode(prompt_text=root.prompt_text, completion_text=full_text, parent=root)
            node.prompt_ids = output.prompt_token_ids
            node.completion_ids = completion.token_ids

            if stop_info["to_stop"]:
                node.mark_terminal(stop_info["reward"], stop_info["stop_reason"])
                extracted_text = full_text[:self.MAX_INVALID_TOKENS_TO_CALC_LOSS_FOR]
                node.completion_text = extracted_text
                node.completion_ids = self.tokenizer.encode(extracted_text)
            else:
                valid_branch_found = True
                thought_count = full_text.count(self.THINK_END_TOKEN) + 1
                if thought_count < self.SPLIT_LEVELS[0]:
                    index = kth_occurrence_from_start(full_text, self.THINK_END_TOKEN, 2)
                elif thought_count <= self.SPLIT_LEVELS[1]:
                    index = kth_occurrence_from_start(full_text, self.THINK_END_TOKEN, 4)
                else:
                    index = kth_occurrence_from_start(full_text, self.THINK_END_TOKEN, 6)

                extracted_text = full_text[:index + len(self.THINK_END_TOKEN)]
                node.completion_text = extracted_text
                node.completion_ids = self.tokenizer.encode(extracted_text)
                node.structured_reward = self.evaluate_sturcture_reward(extracted_text)

            root.add_child(node)
            first_level_nodes.append(node)
        if not valid_branch_found:
            return []

        if thought_count <= self.SPLIT_LEVELS[0]:
            self.max_split_depth = self.SPLIT_COUNTES[0]
        elif thought_count <= self.SPLIT_LEVELS[1]:
            self.max_split_depth = self.SPLIT_COUNTES[1]
        elif thought_count <= self.SPLIT_LEVELS[2]:
            self.max_split_depth = self.SPLIT_COUNTES[2]
        else:
            self.max_split_depth = self.SPLIT_COUNTES[-1]

        return first_level_nodes

    def expand_tree(self, problem: str, final_answer: float) -> TreeNode:
        """
        Build and expand the tree until max_depth is reached.

        Returns a tuple of:
          - all_nodes: a list of all TreeNode instances.
          - terminal_nodes: a list of terminal TreeNode instances.
          - logs: a list of log messages.
        """
        logs: List[str] = []

        terminal_nodes: List[TreeNode] = []
        prompt = self.preprocess_problem(problem)
        root = TreeNode(prompt_text=prompt)


        first_level = self.initial_generation(root)
        if not first_level:
            logs.append("No valid branch found in initial generation.")
            return root

        current_depth = 1

        while current_depth <= self.max_split_depth:
            active_nodes = [node for node in self.get_all_nodes(root) if node.depth == current_depth and not node.is_terminal()]
            if not active_nodes:
                break

            batch_prompts: List[str] = []
            mapping: List[Tuple[TreeNode, int]] = []
            for node in active_nodes:
                splits = self.decide_split(node)

                node.next_split = splits
                if splits > 0:
                    mapping.append((node, splits))
                    prompt_for_generation = node.prompt_text + node.completion_text
                    batch_prompts.extend([prompt_for_generation] * splits)
                elif not node.is_terminal():
                    node.depth += 1

            current_depth += 1
            if not mapping:
                continue

            sampling = self.mid_to_end_sampling_params if current_depth == self.max_split_depth else self.think_sampling_params
            outputs = self.llm.generate(batch_prompts, sampling)
            comp_idx = 0

            for parent, splits in mapping:
                for _ in range(splits):
                    output = outputs[comp_idx]
                    comp_idx += 1
                    completion = output.outputs[0]
                    text = completion.text
                    child = TreeNode(prompt_text=parent.prompt_text + parent.completion_text, completion_text=text, parent=parent)
                    child.prompt_ids = output.prompt_token_ids
                    child.completion_ids = completion.token_ids
                    stop_info = self.handle_stop_conditions(completion, text, final_answer, initial=False)
                    child.completion_text = stop_info["text"]
                    child.structured_reward = self.evaluate_sturcture_reward(stop_info["text"])

                    if stop_info["to_stop"]:
                        child.structured_reward =0 if stop_info['stop_reason']==StopReason.LENGTH  else self.evaluate_sturcture_reward(stop_info["text"])
                        child.mark_terminal(stop_info["reward"], stop_info["stop_reason"])
                    parent.add_child(child)


        # Propagate rewards upward from terminal nodes.
        for node in self.get_all_nodes(root):
            if node.is_terminal():
                node.propagate_reward(node.reward)
        for node in self.get_all_nodes(root):
            if not node.is_terminal():
                node.reward = np.mean(node.rewards)
            node.reward += node.structured_reward
            if len(node.completion_ids) > self.MAX_INVALID_TOKENS_TO_CALC_LOSS_FOR:
                node.completion_ids = node.completion_ids[:self.MAX_INVALID_TOKENS_TO_CALC_LOSS_FOR]
                node.truncated = True
                print('TRUNCATED!!!\n\n')
                node_as_dict = node.to_dict()
                print({k:node_as_dict.get(k) for k in ['prompt_text', 'completion_text', 'state', 'truncated','reward','rewards','depth','stop_reason', 'next_split']})
                print('TO')
                print(self.tokenizer.decode(node.completion_ids))

        json.dump(root.to_dict(),open(os.path.join(PATH_TO_SAVE_DATA,str(time.time())),'w'))

        return root

# ---------------------- Helper Functions ---------------------------
def kth_occurrence_from_end(s: str, sub: str, k: int) -> int:
    """
    Return the index of the k-th occurrence of `sub` in `s` counting from the end.
    Returns -1 if not found.
    """
    pos = len(s)
    for _ in range(k):
        pos = s.rfind(sub, 0, pos)
        if pos == -1:
            return -1
    return pos

def kth_occurrence_from_start(s: str, sub: str, k: int) -> int:
    """
    Return the index of the k-th occurrence of `sub` in `s` counting from the start.
    Returns -1 if the substring does not occur k times.
    """
    pos = -1
    for _ in range(k):
        pos = s.find(sub, pos + 1)
        if pos == -1:
            return -1
    return pos

