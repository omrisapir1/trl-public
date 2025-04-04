import json
import os.path
import time
import re
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from vllm import SamplingParams, LLM

# ---------------------- Node and State Definitions ---------------------------
class NodeState(Enum):
    EXPLORING = 1
    ANSWERING = 2
    TERMINAL = 3
    INVALID = 4

class StopReason(Enum):
    LENGTH = 1
    INVALID_STRUCTURE = 2
    ANSWER_END_TOKEN = 3

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
        self._remove_logs()

    def _remove_logs(self):
        if not os.path.exists(PATH_TO_SAVE_DATA):
            os.makedirs(PATH_TO_SAVE_DATA)
        for cur_f in os.listdir(PATH_TO_SAVE_DATA):
            try:
                os.remove(os.path.join(PATH_TO_SAVE_DATA, cur_f))
            except:
                pass

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
        }

    def add_child(self, child_node: 'TreeNode'):
        child_node.parent = self
        self.children.append(child_node)

    def mark_terminal(self, reward: float, stop_reason: StopReason):
        self.state = NodeState.TERMINAL
        self.reward = reward
        self.stop_reason = stop_reason

    def mark_invalid(self):
        self.state = NodeState.INVALID
        self.reward = 0.0

    def is_terminal(self) -> bool:
        return self.state in {NodeState.TERMINAL, NodeState.INVALID}

    def propagate_reward(self, reward: float):
        self.rewards.append(reward)
        if self.parent:
            self.parent.propagate_reward(reward)

    def compute_final_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def get_full_text(self) -> str:
        """
        Returns the combined text (prompt and completion).
        """
        return self.prompt_text + self.completion_text

# ---------------------- TreeOfThoughts Class ---------------------------
class TreeOfThoughts:
    # Configuration constants
    THINK_END_TOKEN = '</think>'
    THINK_START_TOKEN = '<think>'
    ANSWER_START_TOKEN = '<answer>'
    ANSWER_END_TOKEN = '</answer>'
    END_OF_TEXT_ID_TOKEN = 151643

    MAX_THINK_TOKENS = 512
    MAX_FIRST_ANS_TOKENS = 2548

    CORRECT_STRUCTURE_REWARD = 0.1
    CORRECT_FLOAT_REWARD = 0.1
    FIRST_SPLIT_COUNT = 4
    MIN_THINK_TAG_SPLIT = 1
    LAST_SPLIT = 4

    def __init__(self, llm, max_depth: int = 9, max_split_depth: int = 34):
        self.llm = llm
        self.max_depth = max_depth
        self.max_split_depth = max_split_depth
        self.tokenizer = self.llm.get_tokenizer()

        # Define sampling parameters for different generation stages.
        self.think_sampling_params = SamplingParams(
            temperature=0.9,
            max_tokens=self.MAX_THINK_TOKENS,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            skip_special_tokens=False,
            stop=[self.THINK_END_TOKEN, self.ANSWER_END_TOKEN, self.ANSWER_START_TOKEN],
            n=1,
            include_stop_str_in_output=True,
        )
        self.final_sampling_params = SamplingParams(
            temperature=0.9,
            max_tokens=self.MAX_THINK_TOKENS,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            skip_special_tokens=False,
            stop=[self.ANSWER_END_TOKEN],
            n=1,
            include_stop_str_in_output=True,
        )
        self.first_answer_params = SamplingParams(
            temperature=0.9,
            max_tokens=self.MAX_FIRST_ANS_TOKENS,
            top_p=1.0,
            top_k=50,
            repetition_penalty=1.0,
            skip_special_tokens=False,
            stop=[self.ANSWER_START_TOKEN, self.ANSWER_END_TOKEN],
            n=1,
            include_stop_str_in_output=True,
        )

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
        return prompt + self.THINK_START_TOKEN + self.THINK_END_TOKEN

    def decide_split(self, node: TreeNode) -> int:
        """
        Decide the number of child branches to generate.
        """
        if node.is_terminal():
            return 0
        elif node.next_split is not None:
            return node.next_split
        else:
            return 1

    def get_all_nodes(self, node: 'TreeNode') -> List[TreeNode]:
        nodes = [node]
        for n in node.children:
            nodes += self.get_all_nodes(n)
        return nodes

    def handle_stop_conditions(self, completion: Any, text: str, prompt_text: str,
                               numerical_label: Optional[float] = None, initial: bool = False, is_answering: bool = False) -> Dict[str, Optional[Any]]:
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

        if completion.finish_reason == 'length' or completion.stop_reason == self.END_OF_TEXT_ID_TOKEN or completion.stop_reason is None:
            result["to_stop"] = True
            result["reward"] = 0
            result["stop_reason"] = StopReason.LENGTH if completion.finish_reason == 'length' else StopReason.INVALID_STRUCTURE
            return result

        if initial and (self.ANSWER_END_TOKEN in text or (completion.stop_reason is not None and completion.stop_reason != self.ANSWER_START_TOKEN)):
            result["to_stop"] = True
            result["reward"] = 0
            result["stop_reason"] = StopReason.LENGTH if completion.finish_reason == 'length' else StopReason.INVALID_STRUCTURE
            return result
        if is_answering and (completion.stop_reason != self.ANSWER_END_TOKEN or any([t in text for t in [self.THINK_END_TOKEN, self.ANSWER_START_TOKEN, self.THINK_START_TOKEN]])):
            result["to_stop"] = True
            result["reward"] = 0
            result["stop_reason"] = StopReason.INVALID_STRUCTURE
            return result

        if completion.stop_reason == self.ANSWER_END_TOKEN:
            result["to_stop"] = True
            result["stop_reason"] = StopReason.ANSWER_END_TOKEN
            if self.ANSWER_START_TOKEN in text or self.ANSWER_START_TOKEN in prompt_text:
                reward = self.evaluate_solution(text, numerical_label) + self.CORRECT_STRUCTURE_REWARD if numerical_label is not None else self.CORRECT_STRUCTURE_REWARD
                result["reward"] = reward
            else:
                result["reward"] = 0
            return result
        if completion.stop_reason == self.ANSWER_START_TOKEN:
            result["next_split"] = self.LAST_SPLIT
            return result
        if completion.stop_reason == self.THINK_END_TOKEN:
            return result

        return result

    def evaluate_solution(self, text: str, numerical_label: float) -> float:
        """
        Evaluate the solution by extracting a numerical answer from a \boxed{...} pattern.
        """

        total_reward = 0.0
        matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        if matches:
            extracted = re.sub(r'[^0-9.-]', '', matches[-1]).strip()
            if extracted.replace('.', '', 1).replace('-', '', 1).isdigit():
                try:
                    value = float(extracted)
                    total_reward += self.CORRECT_FLOAT_REWARD
                    total_reward += 1 if value == float(numerical_label) else 0
                    return total_reward
                except ValueError:
                    return total_reward
        return total_reward

    def initial_generation(self, root: TreeNode) -> List[TreeNode]:
        """
        Generate the first set of child nodes from the root.
        """
        full_prompt = root.prompt_text
        prompts = [full_prompt] * self.FIRST_SPLIT_COUNT
        outputs = self.llm.generate(prompts, self.first_answer_params)
        first_level_nodes = []
        valid_branch_found = False

        for output in outputs:
            completion = output.outputs[0]
            full_text = completion.text
            stop_info = self.handle_stop_conditions(completion, full_text, root.prompt_text, initial=True)

            # Create a node with separated prompt and completion.
            node = TreeNode(prompt_text=root.prompt_text, completion_text=full_text, parent=root)
            node.prompt_ids = output.prompt_token_ids
            node.completion_ids = completion.token_ids

            if stop_info["to_stop"]:
                node.mark_terminal(stop_info["reward"], stop_info["stop_reason"])
            else:
                valid_branch_found = True
                thought_count = full_text.count(self.THINK_END_TOKEN)

                node.state = NodeState.ANSWERING
                node.next_split = stop_info["next_split"]
                #
                # if thought_count:
                #
                #     index = kth_occurrence_from_end(full_text, self.THINK_START_TOKEN + self.THINK_END_TOKEN,
                #                                     max(int(np.ceil(thought_count / 2)), self.MIN_THINK_TAG_SPLIT))
                #
                #     if index != -1:
                #         index += len(self.THINK_START_TOKEN + self.THINK_END_TOKEN)
                #         extracted_text = full_text[:index]
                #         node.completion_text = extracted_text
                #         node.completion_ids = self.tokenizer.encode(extracted_text)
                #     else:
                #         node.state = NodeState.ANSWERING
                #         node.next_split = stop_info["next_split"]
            root.add_child(node)
            first_level_nodes.append(node)
        return first_level_nodes if valid_branch_found else []

    def expand_tree(self, problem: str, numerical_label: float) -> TreeNode:
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
        while current_depth < self.max_depth:
            active_nodes = [node for node in self.get_all_nodes(root) if node.depth == current_depth and not node.is_terminal()]
            if not active_nodes:
                break

            batch_prompts: List[str] = []
            mapping: List[Tuple[TreeNode, int]] = []
            for node in active_nodes:
                if node.depth == self.max_depth:
                    terminal_nodes.append(node)
                    continue
                splits = self.decide_split(node)
                node.next_split = splits
                if splits > 0:
                    mapping.append((node, splits))
                    prompt_for_generation = node.prompt_text + node.completion_text
                    batch_prompts.extend([prompt_for_generation] * splits)
                else:
                    terminal_nodes.append(node)

            if not mapping:
                break

            sampling = self.final_sampling_params if current_depth == (self.max_depth - 1) else self.think_sampling_params
            outputs = self.llm.generate(batch_prompts, sampling)
            comp_idx = 0

            for parent, splits in mapping:
                is_answering = parent.state == NodeState.ANSWERING
                # If the decided split is 1, extend the parent node rather than adding a new node.
                if splits == 1:
                    output = outputs[comp_idx]
                    comp_idx += 1
                    completion = output.outputs[0]
                    text = completion.text
                    stop_info = self.handle_stop_conditions(completion, text, parent.prompt_text, numerical_label, initial=False, is_answering=is_answering)

                    # Extend parent's completion text and token ids.
                    parent.completion_text += stop_info["text"]
                    parent.completion_ids += completion.token_ids
                    parent.depth += 1  # Increase parent's depth to reflect extension.
                    if stop_info["to_stop"]:
                        parent.mark_terminal(stop_info["reward"], stop_info["stop_reason"])
                    if stop_info["next_split"] is not None:
                        parent.next_split = stop_info["next_split"]
                        parent.state = NodeState.ANSWERING
                else:
                    for _ in range(splits):
                        output = outputs[comp_idx]
                        comp_idx += 1
                        completion = output.outputs[0]
                        text = completion.text
                        child = TreeNode(prompt_text=parent.prompt_text + parent.completion_text, completion_text=text, parent=parent)
                        child.prompt_ids = output.prompt_token_ids
                        child.completion_ids = completion.token_ids
                        stop_info = self.handle_stop_conditions(completion, text, parent.prompt_text + parent.completion_text, numerical_label, initial=False, is_answering=is_answering)

                        child.completion_text = stop_info["text"]
                        if stop_info["to_stop"]:
                            child.mark_terminal(stop_info["reward"], stop_info["stop_reason"])

                        if stop_info["next_split"] is not None:
                            child.state = NodeState.ANSWERING
                            child.next_split = stop_info["next_split"]

                        parent.add_child(child)

            current_depth += 1

        # Propagate rewards upward from terminal nodes.
        for node in self.get_all_nodes(root):
            if node.is_terminal():
                node.propagate_reward(node.reward)
        for node in self.get_all_nodes(root):
            if not node.is_terminal():
                node.reward = np.mean(node.rewards)

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
