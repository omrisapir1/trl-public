import os
import asyncio, json, uuid, re, math, time, shutil, numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Iterable

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

from .extract_answer import extract_final_answer, math_equal

# ──────────────────────────────────────────────────────────────────────────────
# constants / hyper‑params
# ──────────────────────────────────────────────────────────────────────────────
MAX_STREAMS        = 128
TAU                = 0.4     # entropy trigger for splits *after* first‑pass
TEMP               = 0.99
TOP_P, TOP_K       = 0.9, 50
REP_PENALTY        = 1.1
LOGPROBS_K         = 20
MAX_TOKENS_GEN     = 4_000
MIN_SPLIT_TOKENS   = 10

SPLITABLE_TOKENS   = {"\n", "!", ".", "?"}   # thought separators
MAX_DEPTH_SPLIT    = 7       # same as MAX_SPLIT in discussion

SAVE_DIR           = Path("training_data_entropy_vllm")
SAVE_DIR.mkdir(exist_ok=True)
# clean directory each run (dev convenience)
for child in SAVE_DIR.iterdir():
    if child.is_dir(): shutil.rmtree(child)
    else: child.unlink(missing_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# node data model
# ──────────────────────────────────────────────────────────────────────────────
class NodeState(Enum):
    EXPLORING = auto()
    TERMINAL  = auto()

class StopReason(Enum):
    DONE = auto()

def _collapse_separators(txt: str) -> Tuple[bool,int,str]:
    """Returns (is_break,new_breaks,last_char) after collapsing runs of separators+ws."""
    if not txt:
        return False,0,""
    i = len(txt)-1
    cnt = 0
    while i>=0 and txt[i].isspace():
        i -= 1
    while i>=0 and txt[i] in SPLITABLE_TOKENS:
        cnt += 1; i -= 1
        while i>=0 and txt[i].isspace():
            i -= 1
    return cnt>0, cnt, txt[-1]

def _token_entropy(lps: List[float]) -> float:
    p = np.exp(lps); p /= p.sum()
    return float(-(p*np.log(p)).sum())

@dataclass
class TreeNode:
    prompt_ids: List[int]
    depth: int = 0
    parent: Optional["TreeNode"] = None

    prompt_text: str = ""
    completion_text: str = ""
    completion_ids: List[int] = field(default_factory=list)
    children: List["TreeNode"] = field(default_factory=list)

    # runtime flags
    first_pass: bool = False              # streaming immunity flag
    thought_quota: Optional[int] = None   # for logging/debug

    state: NodeState = NodeState.EXPLORING
    reward: Optional[float] = None
    rewards: List[float] = field(default_factory=list)
    stop_reason: Optional[StopReason] = None

    # helpers ----------------------------------------------------------
    def add_child(self, c:"TreeNode") -> None: self.children.append(c)

    def propagate_reward(self,r:float):
        self.rewards.append(r)
        if self.parent: self.parent.propagate_reward(r)

    def compute_final_reward(self)->float:
        return sum(self.rewards)/len(self.rewards) if self.rewards else 0.0

    def traverse(self):
        stack=[self]
        while stack:
            n=stack.pop(); yield n; stack.extend(n.children)

# ──────────────────────────────────────────────────────────────────────────────
class TreeOfThoughtsEntropyVLLM:
    """Two‑phase ToT with initial token split + thought‑quota truncation."""

    def __init__(self, *, engine: AsyncLLMEngine, tokenizer: AutoTokenizer):
        self.engine, self.tokenizer = engine, tokenizer
        self.sem  = asyncio.Semaphore(MAX_STREAMS)

    # chat prompt ------------------------------------------------------
    def _prompt(self, problem:str)->List[int]:
        return self.tokenizer.apply_chat_template([
            {"role":"user","content":problem}], tokenize=True,
            add_generation_prompt=True, continue_final_message=False)

    # evaluation -------------------------------------------------------
    def _reward(self, txt:str, ans:str)->float:
        try: return int(math_equal(extract_final_answer(txt), ans))
        except: return 0

    # ------------------------------------------------------------------
    async def expand_tree(self, problem:str, answer:str)->TreeNode:
        root = TreeNode(self._prompt(problem))
        await self._seed_first_two_children(root, answer)
        # wait all spawned tasks
        if hasattr(self,"_tasks") and self._tasks:
            while (pending:=[t for t in self._tasks if not t.done()]):
                await asyncio.gather(*pending)
        # propagate rewards
        for n in root.traverse():
            if n.state is NodeState.TERMINAL: n.propagate_reward(n.reward or 0)
        for n in root.traverse():
            if n.state is not NodeState.TERMINAL: n.reward=n.compute_final_reward()
        SAVE_DIR.joinpath(f"{time.time()}.json").write_text(json.dumps(self._to_dict(root),indent=2))
        return root

    # ------------------------------------------------------------------
    async def _seed_first_two_children(self, root:TreeNode, answer:str):
        """Generate **one token**, branch into original+alt children, fully generate them."""
        params = SamplingParams(temperature=TEMP,top_p=TOP_P,top_k=TOP_K,
                                repetition_penalty=REP_PENALTY,max_tokens=1,logprobs=LOGPROBS_K)
        async for chunk in self.engine.generate(self.tokenizer.decode(root.prompt_ids), params, request_id=str(uuid.uuid4())):
            out = chunk.outputs[0]
            tok_id = out.token_id; tok_lp = out.logprobs[-1][tok_id].logprob
            top = {tid:e.logprob for tid,e in out.logprobs[-1].items()}
            alt_ids,lps = zip(*[(t,lp) for t,lp in top.items() if t!=tok_id])
            alt_id = int(np.random.choice(alt_ids, p=np.exp(lps)/np.exp(lps).sum()))
            break
        # create children
        for forced in (tok_id, alt_id):
            child = TreeNode(root.prompt_ids+[forced], depth=1, parent=root, first_pass=True)
            root.add_child(child)
            self._tasks = getattr(self,"_tasks",[])
            self._tasks.append(asyncio.create_task(self._spawn(child, answer)))

    # ------------------------------------------------------------------
    async def _spawn(self, node:TreeNode, answer:str):
        async with self.sem:
            params = SamplingParams(temperature=TEMP,top_p=TOP_P,top_k=TOP_K,
                                    repetition_penalty=REP_PENALTY,max_tokens=MAX_TOKENS_GEN,logprobs=LOGPROBS_K)
            prefix_text = self.tokenizer.decode(node.prompt_ids)
            thoughts_seen = 0; quota_reached=False
            stream_restart = False
            async for chunk in self.engine.generate(prefix_text, params, request_id=str(uuid.uuid4())):
                out = chunk.outputs[0]
                # append token text/ids
                node.completion_ids.append(out.token_id)
                node.completion_text += out.text
                # separator collapse
                is_break,new_breaks,last_char = _collapse_separators(out.text)
                if is_break and not quota_reached: thoughts_seen += new_breaks
                # first‑pass logic ------------------------------------------------
                if node.first_pass:
                    if chunk.finish_reason is not None:  # reached </s>
                        total_thoughts = max(1, thoughts_seen)
                        quota = math.floor(max(total_thoughts-MAX_DEPTH_SPLIT,0)/2)
                        quota = max(1, quota)
                        node.thought_quota = quota
                        # truncate if needed
                        if total_thoughts>MAX_DEPTH_SPLIT and quota<total_thoughts:
                            node.completion_ids, node.completion_text = self._truncate_to_thoughts(node.completion_ids, quota)
                        # flip flag and restart streaming from new prefix
                        node.first_pass=False
                        await self._resume_stream(node, answer)
                        return
                    continue  # still in first pass → skip entropy logic

                # entropy split logic after first pass ------------------------
                top = {tid:e.logprob for tid,e in out.logprobs[-1].items()} if out.logprobs else {}
                if (
                    is_break and len(top)>1 and (H:=_token_entropy(list(top.values())))>TAU and
                    node.depth<MAX_DEPTH_SPLIT and len(node.completion_ids)>MIN_SPLIT_TOKENS
                ):
                    await self._entropy_branch(node, out, top, answer)
                    return
            # stream finished normally --------------------------------------
            node.state=NodeState.TERMINAL
            node.stop_reason=StopReason.DONE
            node.reward=self._reward(node.completion_text, answer)

    # -------- resume streaming after truncation ---------------------------
    async def _resume_stream(self, node:TreeNode, answer:str):
        # simply call _spawn again (without first_pass flag)
        await self._spawn(node, answer)

    # -------- entropy branch ---------------------------------------------
    async def _entropy_branch(self,node:TreeNode,out,top,answer):
        tok_id = node.completion_ids.pop()   # remove last token from parent
        node.completion_text = self.tokenizer.decode(node.completion_ids)
        next_prompt_ids = node.prompt_ids + node.completion_ids
        cand_ids,cand_lps = zip(*[(t,lp) for t,lp in top.items() if t!=tok_id])
        probs = np.exp(cand_lps); probs/=probs.sum(); alt_id=int(np.random.choice(cand_ids,p=probs))
        for forced in (tok_id,alt_id):
            child = TreeNode(next_prompt_ids+[forced], depth=node.depth+1, parent=node)
            node.add_child(child)
            self._tasks=getattr(self,"_tasks",[])
            self._tasks.append(asyncio.create_task(self._spawn(child, answer)))

    # -------- truncate helper -------------------------------------------
    def _truncate_to_thoughts(self, ids:List[int], quota:int)->Tuple[List[int],str]:
        txt=self.tokenizer.decode(ids)
        breaks=0; cut_idx=len(txt)
        for m in re.finditer(r"[!.?\n]+", txt):
            breaks+=1
            if breaks==quota:
                cut_idx = m.end()
                break
        new_txt = txt[:cut_idx]
        new_ids = self.tokenizer.encode(new_txt, add_special_tokens=False)
        return new_ids, new_txt

    # -------- tree traversal -------------------------------------------
    def _to_dict(self,root:TreeNode)->Dict[str,Any]:
        return root.to_dict()
