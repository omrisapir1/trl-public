
import os
import asyncio, json, uuid, re, numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
try:
    from .extract_answer import extract_final_answer, math_equal
except:
    from extract_answer import extract_final_answer, math_equal
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine


MAX_STREAMS = 128
TAU = 0.6#1#0.6  # threshold on EMA entropy
# TAU = [1.1, 1.5 , 1.4 , 1.3, 0.9 ,0.9, 1.1]
TEMP = 0.6
TOP_P = 0.85
TOP_K = 20
REP_PENALTY = 1.1
LOGPROBS_K = 20
MAX_TOKENS = 1300
MIN_SPLIT_TOKENS = 60
LAST_SPLIT_MIN_CHARS = 150



from pathlib import Path
import shutil

SAVE_DIR = Path("training_data_entropy_vllm")
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)


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
            'answer': self.answer if hasattr(self, 'answer') else None,
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
                                 remove_last_token: bool, new_completion_ids=None) -> int:
        last_token_id = None
        init_addition_tokens = []
        if take_one_from_prompt:
            init_addition_tokens.append(node.prompt_ids[-1])
            node.prompt_ids = node.prompt_ids[:-1]

        node.completion_ids = init_addition_tokens + list(output.token_ids)
        if remove_last_token:
            last_token_id = node.completion_ids[-1]
            node.completion_ids = node.completion_ids[:-1]
        elif new_completion_ids:
            node.completion_ids = new_completion_ids

        node.prompt_text = self.tokenizer.decode(node.prompt_ids)
        node.completion_text = self.tokenizer.decode(node.completion_ids)
        return last_token_id

    # ----------------------------------------------------------- main entry ---
    async def expand_tree(self, problem: str, answer: str) -> TreeNode:
        self.cur_split_count = 0
        root = TreeNode(self._prompt(problem))
        await self._spawn(root, answer, None)
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
        root.answer = answer
        SAVE_DIR.joinpath(f"{time.time()}.json").write_text(json.dumps(root.to_dict(), indent=2))

        return root

    # ---------------------------------------------------------------- spawn ---
    async def _spawn(self, node: TreeNode, answer: str, after_last_split: bool = False):
        """Core recursive worker.  Streams one token at a time even if vLLM
        delivers multi-token chunks."""
        async with self.sem:
            params = SamplingParams(
                temperature=TEMP,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=REP_PENALTY,
                max_tokens=MAX_TOKENS,
                logprobs=LOGPROBS_K,
            )

            prompt_text = self.tokenizer.decode(node.prompt_ids)
            token_acc: List[int] = []  # tokens revealed so far in THIS call
            at_splitable_token = False

            async for chunk in self.engine.generate(
                    prompt_text, params, request_id=str(uuid.uuid4())
            ):
                if after_last_split:
                    continue

                out = chunk.outputs[0]
                new_cnt = getattr(out, "num_generated_tokens", len(out.token_ids))
                start_ix = len(out.token_ids) - new_cnt
                log_seqs = out.logprobs or [{} for _ in out.token_ids]

                # ── iterate per-token ───────────────────────────────────────
                for idx in range(start_ix, len(out.token_ids)):
                    tid = out.token_ids[idx]
                    lp_dict = log_seqs[idx] if idx < len(log_seqs) else {}
                    token_txt = self.tokenizer.decode([tid])
                    token_acc.append(tid)

                    # ---- entropy on THIS token ----------------------------
                    raw_H = 0.0
                    if lp_dict:
                        vals = []
                        for v in lp_dict.values():  # v may be obj or float
                            lp = v.logprob if hasattr(v, "logprob") else float(v)
                            vals.append(-99.0 if np.isinf(lp) else lp)
                        p = np.exp(vals - np.max(vals))
                        p /= p.sum()
                        raw_H = float(-(p * np.log(p)).sum())

                    # ---- split decision -----------------------------------
                    split_now = (
                                        lp_dict
                                        and node.depth < MAX_DEPTH_SPLIT
                                        and raw_H > TAU
                                        and at_splitable_token
                                        and token_txt not in SPLITABLE_TOKENS
                                ) or (node.depth == 0)

                    if split_now:
                        # 1 — update parent (remove hi-entropy token)
                        fake_out = types.SimpleNamespace(
                            token_ids=token_acc.copy(),
                            text=self.tokenizer.decode(token_acc),
                            logprobs=[lp_dict],
                        )
                        tok_id = self._update_node_with_output(
                            node,
                            fake_out,
                            take_one_from_prompt=(node.depth != 0),
                            remove_last_token=True,
                        )
                        next_prompt_ids = node.prompt_ids + node.completion_ids

                        # 2 — sample an alternative token for the sibling
                        cand_pairs = [
                                         (t, (v.logprob if hasattr(v, "logprob") else float(v)))
                                         for t, v in lp_dict.items()
                                         if t == tok_id
                                     ] or [(tok_id, 0.0)]
                        cand_ids, cand_lps = zip(*cand_pairs)
                        probs = np.exp(np.array(cand_lps) / TEMP)
                        probs /= probs.sum()
                        alt_tid = int(np.random.choice(cand_ids, p=probs))

                        # 3 — spawn children and stop this stream
                        for forced in (tok_id, alt_tid):
                            child = TreeNode(
                                next_prompt_ids + [forced],
                                depth=node.depth + 1,
                                parent=node,
                            )
                            node.add_child(child)
                            self._tasks.append(
                                asyncio.create_task(self._spawn(child, answer))
                            )
                        return  # parent stream ends after split

                    # remember if this token makes future split legal
                    at_splitable_token = token_txt in SPLITABLE_TOKENS

            # ── generation finished without split ──────────────────────────
            take_one_from_prompt = False
            gen_text = self.tokenizer.decode(token_acc)

            if not after_last_split:
                cut = self._last_occurrence(
                    gen_text[:-LAST_SPLIT_MIN_CHARS], SPLITABLE_TOKENS
                )
                new_comp_ids = self.tokenizer.encode(gen_text[: cut + 1])

                if cut != -1 and len(new_comp_ids) > MIN_SPLIT_TOKENS:
                    fake_out = types.SimpleNamespace(
                        token_ids=new_comp_ids,
                        text=self.tokenizer.decode(new_comp_ids),
                        logprobs=[log_seqs[-1] if log_seqs else {}],
                    )
                    self._update_node_with_output(
                        node,
                        fake_out,
                        take_one_from_prompt=(node.depth != 0),
                        remove_last_token=False,
                        new_completion_ids=new_comp_ids,
                    )
                    next_prompt_ids = node.prompt_ids + node.completion_ids
                    for _ in range(4):
                        child = TreeNode(
                            next_prompt_ids, depth=node.depth + 1, parent=node
                        )
                        node.add_child(child)
                        self._tasks.append(
                            asyncio.create_task(
                                self._spawn(child, answer, after_last_split=True)
                            )
                        )
                    return
                else:
                    # back-off: push token back to parent and retry
                    take_one_from_prompt = True
                    next_prompt_ids = node.prompt_ids[:-1]
                    child = TreeNode(next_prompt_ids, depth=node.depth, parent=node.parent)
                    node.parent.add_child(child)
                    self._tasks.append(
                        asyncio.create_task(
                            self._spawn(child, answer, after_last_split=True)
                        )
                    )

            # ---- mark node as terminal -----------------------------------
            final_out = types.SimpleNamespace(
                token_ids=token_acc.copy(),
                text=self.tokenizer.decode(token_acc),
                logprobs=[log_seqs[-1] if log_seqs else {}],
            )
            self._update_node_with_output(
                node,
                final_out,
                take_one_from_prompt=take_one_from_prompt,
                remove_last_token=False,
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