
import os
import asyncio, json, uuid, re, numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from .extract_answer import extract_final_answer, math_equal
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine


MAX_STREAMS = 128
TAU = 1  # threshold on EMA entropy
# TAU = [1.1, 1.5 , 1.4 , 1.3, 0.9 ,0.9, 1.1]
TEMP = 0.99
TOP_P = 0.9
TOP_K = 50
REP_PENALTY = 1.1
LOGPROBS_K = 20
MAX_TOKENS_GEN = 4000
MIN_SPLIT_TOKENS = 30
LAST_SPLIT_MIN_TOKENS = 30


SAVE_DIR = Path("training_data_entropy_vllm");
from pathlib import Path
import shutil

SAVE_DIR = Path("training_data_entropy_vllm")

# Iterate over everything directly inside the directory
for child in SAVE_DIR.iterdir():
    if child.is_file() or child.is_symlink():
        child.unlink()           # remove file or symlink
    else:
        shutil.rmtree(child)


SPLITABLE_TOKENS = {'\n', '!', '.', '?'}
MAX_DEPTH_SPLIT = 6


class NodeState(Enum):
    EXPLORING = auto()
    TERMINAL = auto()


class StopReason(Enum):
    DONE = auto()


@dataclass
class TreeNode:
    prompt_ids: List[int]
    depth: int = 0
    parent: Optional["TreeNode"] = None

    prompt_text: str = ""
    completion_text: str = ""
    completion_ids: List[int] = field(default_factory=list)
    children: List["TreeNode"] = field(default_factory=list)

    # runtime signals
    ema_entropy: list[float] = field(default_factory=list)
    state: NodeState = NodeState.EXPLORING
    reward: Optional[float] = None
    rewards: List[float] = field(default_factory=list)
    stop_reason: Optional[StopReason] = None

    def add_child(self, c: "TreeNode") -> None:
        self.children.append(c)

    def propagate_reward(self, r: float):
        self.rewards.append(r)
        if self.parent:
            self.parent.propagate_reward(r)

    def compute_final_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def traverse(self):
        stack = [self]
        while stack:
            n = stack.pop();
            yield n;
            stack.extend(n.children)

    # avg advantage for immediate children
    def local_advantage(self):
        if not self.children: return 0.0, 0
        rewards = [c.reward for c in self.children]
        tok_cnt = sum(len(c.completion_ids) for c in self.children)
        return (max(rewards) - float(np.mean(rewards))), tok_cnt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "prompt_text": self.prompt_text,
            "completion_text": self.completion_text,
            "state": self.state.name,
            "reward": self.reward,
            "rewards": self.rewards,
            "depth": self.depth,
            "children": [c.to_dict() for c in self.children],
        }


# ──────────────────────────────────────────────────────────────────────────────
class TreeOfThoughtsEntropyVLLM:
    """EMA‑smoothed entropy tree search driven by AsyncLLMEngine."""

    def __init__(self, *, engine: AsyncLLMEngine, tokenizer: AutoTokenizer) -> None:
        self.engine, self.tokenizer = engine, tokenizer
        self.sem = asyncio.Semaphore(MAX_STREAMS)

        SAVE_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------- helpers ----
    def _prompt(self, problem: str) -> List[int]:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=True,
            add_generation_prompt=True,
            continue_final_message=False,
        )

    def evaluate_solution(self, text: str, final_answer: str) -> float:
        """
        Evaluate the solution by extracting a numerical answer from a \boxed{...} pattern.
        """
        final_prediction = extract_final_answer(text)
        try:
            return int(math_equal(final_prediction, final_answer))
        except:
            return 0

    def _update_node_with_output(self, node: TreeNode, output: Any, take_one_from_prompt: bool,
                                 remove_last_token: bool, last_split=False) -> int:
        last_token_id = None
        init_addition_tokens = []
        if take_one_from_prompt:
            init_addition_tokens.append(node.prompt_ids[-1])
            node.prompt_ids = node.prompt_ids[:-1]

        node.completion_ids = init_addition_tokens + list(output.token_ids)
        if remove_last_token:
            last_token_id = node.completion_ids[-1]
            node.completion_ids = node.completion_ids[:-1]
        elif last_split:
            last_index_splitable_char = last_occurrence(output.text[:LAST_SPLIT_MIN_TOKENS], SPLITABLE_TOKENS)
            node.completion_ids = self.tokenizer.encode(output.text[:last_index_splitable_char + 1])


        node.prompt_text = self.tokenizer.decode(node.prompt_ids)
        node.completion_text = self.tokenizer.decode(node.completion_ids)
        return last_token_id

    # ----------------------------------------------------------- main entry ---
    async def expand_tree(self, problem: str, answer: str) -> TreeNode:
        self.cur_split_count = 0
        root = TreeNode(self._prompt(problem))
        await self._spawn(root, answer)
        if hasattr(self, "_tasks") and self._tasks:
            await asyncio.gather(*self._tasks)

            while True:
                # keep only tasks that are not finished yet
                pending = [t for t in self._tasks if not t.done()]
                if not pending:
                    break
                # await the current batch; new tasks may be appended meanwhile
                await asyncio.gather(*pending)

        for n in self._all_nodes(root):
            if n.state is NodeState.TERMINAL:
                n.propagate_reward(n.reward or 0.0)
        for n in self._all_nodes(root):
            if n.state is not NodeState.TERMINAL:
                n.reward = n.compute_final_reward()
        SAVE_DIR.joinpath(f"{time.time()}.json").write_text(json.dumps(root.to_dict(), indent=2))
        return root

    # ---------------------------------------------------------------- spawn ---
    async def _spawn(self, node: TreeNode, answer: str, after_last_split=False):
        async with self.sem:
            params = SamplingParams(
                temperature=TEMP,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REP_PENALTY,
                max_tokens=MAX_TOKENS_GEN,
                logprobs=LOGPROBS_K,
            )
            prompt_text = self.tokenizer.decode(node.prompt_ids)
            ema_entropy = []
            at_splitable_token = False
            async for chunk in self.engine.generate(prompt_text, params, request_id=str(uuid.uuid4())):
                if after_last_split:
                    continue
                out = chunk.outputs[0]
                total_tokens = len(out.token_ids)
                if total_tokens < MIN_SPLIT_TOKENS:
                    continue

                # --- entropy --------------------------------------------------
                top = {tid: -99 if np.isinf(e.logprob) else e.logprob for tid, e in out.logprobs[-1].items() } if out.logprobs else {}
                raw_H = 0.0


                if len(top) > 1:


                    p = np.exp(list(top.values()));
                    p /= p.sum()
                    raw_H = float(-(p * np.log(p)).sum())
                    # mvg_avg = np.mean(ema_entropy[-self.window_k:-1]) * (1 - self.alpha_ema) + self.alpha_ema * raw_H
                    # ema_entropy.append(raw_H)



                # --- split decision ------------------------------------------

                # mvg_avg_normalized = mvg_avg / (1 + np.exp(-self.k * (total_tokens - self.t0)))
                # mvg_avg_normalized = mvg_avg * np.sqrt(total_tokens / self.t0)
                # if at_splitable_token:
                #     print(top.values())
                #     print(raw_H)


                if (
                        len(top) > 1 and node.depth < MAX_DEPTH_SPLIT and raw_H > TAU
                        and at_splitable_token
                        # self.cur_split_count < MAX_TOTAL_SPLITS
                ) or (node.depth==0):

                    self.cur_split_count += 1

                    # remove the high‑entropy token from parent, return its id
                    tok_id = self._update_node_with_output(
                        node,
                        out,
                        take_one_from_prompt=(node.depth != 0),
                        remove_last_token=True,
                    )
                    next_prompt_ids = node.prompt_ids + node.completion_ids

                    cand_ids, cand_lps = zip(*[(t, lp) for t, lp in top.items() if t != tok_id])
                    probs = np.exp(np.array(cand_lps) / TEMP);
                    probs /= probs.sum()
                    alt_tid = int(np.random.choice(cand_ids, p=probs))

                    for forced in (tok_id, alt_tid):
                        child = TreeNode(next_prompt_ids + [forced], depth=node.depth + 1, parent=node)
                        node.add_child(child)
                        self._tasks = getattr(self, "_tasks", [])
                        self._tasks.append(asyncio.create_task(self._spawn(child, answer)))
                    return  # stop parent stream
                at_splitable_token = out.text[-1] in SPLITABLE_TOKENS if out.text else False


            if not after_last_split:
                self._update_node_with_output(
                    node,
                    out,
                    take_one_from_prompt=(node.depth != 0),
                    remove_last_token=False,
                    last_split=True,

                )
                next_prompt_ids = node.prompt_ids + node.completion_ids
            
                for _ in range(2):
                    child = TreeNode(next_prompt_ids, depth=node.depth + 1, parent=node)
                    node.add_child(child)
                    self._tasks = getattr(self, "_tasks", [])
                    self._tasks.append(asyncio.create_task(self._spawn(child, answer, after_last_split=True)))
                return


            self._update_node_with_output(
                node,
                out,
                take_one_from_prompt=False,
                remove_last_token=False,  # keep last token

            )
            node.state = NodeState.TERMINAL
            node.stop_reason = StopReason.DONE
            node.reward = self.evaluate_solution(node.completion_text, answer)

    # ------------------------------------------------------------------
    def _all_nodes(self, root: TreeNode) -> List[TreeNode]:
        st, out = [root], []
        while st:
            n = st.pop();
            out.append(n);
            st.extend(n.children)
        return out

def last_occurrence(s: str, chars: set) -> int:
    for i in range(len(s) - 1, -1, -1):
        if s[i] in chars:
            return i
    return -1