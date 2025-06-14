import json
import os
import tempfile
import time
import uuid

from vllm import SamplingParams
from ..extras.vllm_client import VLLMClient
try:
    from .extract_answer import extract_final_answer, math_equal
except:
    from extract_answer import extract_final_answer, math_equal

# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import textwrap
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable, Optional, Sized, Union, List

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler, BatchSampler, RandomSampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from safetensors.torch import save_file
from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..extras.profiling import profiling_context, profiling_decorator
# from ..extras.vllm_client import VLLMClient
# from vllm import LLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from ..import_utils import is_deepspeed_available, is_rich_available, is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig
from .ToT import TreeOfThoughts
from .ToE import TreeOfThoughtsEntropyVLLM
from .utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
import asyncio

# If you plan to run in notebooks, install nest_asyncio once:
# pip install nest_asyncio
import nest_asyncio
nest_asyncio.apply()


def run_async(coro):
    """
    Execute *coro* (a coroutine) and return its result.

    • In a normal Python script     → starts a fresh event loop.
    • In a running event-loop (e.g. Jupyter) → re-uses that loop safely.
    """
    try:
        loop = asyncio.get_running_loop()
        # Already inside an event-loop (Jupyter, web-server …)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No loop running – create one
        return asyncio.run(coro)

if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftConfig, get_peft_model


if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

MAX_TOKENS_TO_CALC_LOSS = 512
class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,

     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)



class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return None when the reward is not applicable to those samples. This is useful for
                  multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns None for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,

    ):
        # Args
        self.model_dir = model
        self._n_gpu = 1
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model,torch_dtype='auto',trust_remote_code=True, **model_init_kwargs)
            from collections import Counter

            # Count dtypes across all parameters
            dtype_counts = Counter(param.dtype for param in model.parameters())
            print(dtype_counts)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Enable gradient checkpointing if requested
        self._n_gpu = 1
        args.__dict__["n_gpu"] = self._n_gpu
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype='auto',trust_remote_code=True, **model_init_kwargs)
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        # self.ref_model = self.ref_model.to('cuda:1')

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.num_completions_to_print = args.num_completions_to_print
        from dataclasses import replace
        self.args = replace(args, _n_gpu=self._n_gpu)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        # possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        # if self.num_generations not in possible_values:
        #     raise ValueError(
        #         f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
        #         f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
        #         f"batch size, the valid values for the number of generations are: {possible_values}."
        #     )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                pass
                self.vllm_client = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
                                model=model.name_or_path,
                                # tensor_parallel_size=2,
                                # device='cuda:1',
                                gpu_memory_utilization=0.5,
                                dtype=torch.bfloat16,
                                max_num_seqs=128,
                                disable_log_stats=True,

                                max_num_batched_tokens=64 * 3500,
                                trust_remote_code=True,

                                # tensor_parallel_size=2,
                                # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                                # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                                # This is particularly useful here because we generate completions from the same prompts.
                                enable_prefix_caching=True,
                                # max_model_len=24000,
                            ))
                self.vllm_client.log_requests = False
                self.log_results_200(skip_first=True)
            # VLLMClient(
                #     args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                # )
                #

                # self.vllm_client = VLLMClient(
                #     args.vllm_server_host, args.vllm_server_port, connection_timeout=args.vllm_server_timeout
                # )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
            self.tree_of_thought = TreeOfThoughtsEntropyVLLM(engine=self.vllm_client, tokenizer=self.tokenizer)
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                bos_token_id=processing_class.bos_token_id,
                eos_token_id=processing_class.eos_token_id,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
                cache_implementation=args.cache_implementation,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        # self.ref_model = self.ref_model.to('cuda:1')
        # self.accelerator.device = 'cuda:0'

    async def run_one(self, prompt: str, request_id: str) -> str:   # ← async
        params = SamplingParams(
            max_tokens=5000,
            temperature=0.0,
            skip_special_tokens=False,
        )
        async for out in self.vllm_client.generate(prompt, params, request_id):
            if out.finished:
                return out.outputs[0].text
        return ""        # should not reach

    # ── 2. batch helper ──────────────────────────────────────────────────────────
    async def pred_prompts(self, prompts_list):
        tasks = [asyncio.create_task(self.run_one(p, f"req-{i}"))
                 for i, p in enumerate(prompts_list)]
        return await asyncio.gather(*tasks)          #   no shutdown here


    def log_results_200(self, skip_first=False):
        if skip_first:
            return

        import pandas as pd, time, json, torch
        from transformers import GenerationConfig

        # helper to wrap problems in chat template
        def _prompt(problem: str) -> str:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )

        # ---------- load test set -------------------------------------------------
        df = pd.read_csv("/workspace/Qwq_32b_awq_greedy_200_test.csv")
        prompt_list = df["problem"].apply(_prompt).tolist()

        # ---------- 1) vLLM -------------------------------------------------------
        preds_vllm = asyncio.run(self.pred_prompts(prompt_list))     # ← await it
        df["pred_vllm"] = preds_vllm
        df["numerical_pred_vllm"] = df["pred_vllm"].apply(extract_final_answer)

        acc_vllm = df.apply(
            lambda r: math_equal(r["numerical_solution"], r["numerical_pred_vllm"]),
            axis=1,
        ).mean()
        tok_avg_vllm = df["pred_vllm"].apply(lambda t: len(self.tokenizer.encode(t))).mean()

        if skip_first:
            print(acc_vllm, tok_avg_vllm)
            return

        # ---------- 2) Transformers ----------------------------------------------
        def _pred_xfmr(prompts, batch_size=16):
            gen_cfg = GenerationConfig(
                max_new_tokens=3000,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            self.model.eval()
            device = next(self.model.parameters()).device
            outs = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                toks = self.tokenizer(batch, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    gen = self.model.generate(**toks, generation_config=gen_cfg)
                prompt_len = toks["input_ids"].shape[1]
                outs.extend(
                    self.tokenizer.batch_decode(gen[:, prompt_len:], skip_special_tokens=False)
                )
            return outs

        preds_xfmr = _pred_xfmr(prompt_list)
        df["pred_xfmr"] = preds_xfmr
        df["numerical_pred_xfmr"] = df["pred_xfmr"].apply(extract_final_answer)

        acc_xfmr = df.apply(
            lambda r: math_equal(r["numerical_solution"], r["numerical_pred_xfmr"]),
            axis=1,
        ).mean()
        tok_avg_xfmr = df["pred_xfmr"].apply(lambda t: len(self.tokenizer.encode(t))).mean()

        # ---------- save & print --------------------------------------------------
        res = {
            "accuracy_vllm": float(acc_vllm),
            "accuracy_xfmr": float(acc_xfmr),
            "tokens_avg_vllm": float(tok_avg_vllm),
            "tokens_avg_xfmr": float(tok_avg_xfmr),
        }
        ts = time.time()
        with open(f"/workspace/eval_{ts:.0f}.json", "w") as f:
            json.dump(res, f, indent=2)

        print(f"[eval @ step {self.state.global_step}] {res}")



    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                     |     GPU 0     |     GPU 1     |     GPU 2    |
        #
        #               global_step   step     <───────>  num_generations=3
        #                                      <───────────> per_device_train_batch_size=4
        #                ▲   0          0      0   0   0   1   1   1   2   2   2   3   3   3  │
        #  grad_accum=3  │   0          1      4   4   4   5   5   5   6   6   6   7   7   7  │ Generate completions for each prompt
        #                ▼   0          2      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    1          3      0   0   0   1   1   1   2   2   2   3   3   3  │ The sampled prompts are the same as in the first iteration
        #                    1          4      4   4   4   5   5   5   6   6   6   7   7   7  │ Reuse the completions (here, once, because num_iterations=2)
        #                    1          5      8   8   8   9   9   9  10  10  10  11  11  11  │
        #
        #                    2          6     12  12  12  13  13  13  14  14  14  15  15  15
        #                    2          7     16  16  16  17  17  17  18  18  18  19  19  19
        #                    2          8     20  20  20  21  21  21  22  22  22  23  23  23
        #                                          ...
        # return BatchSampler(
        #     sampler=RandomSampler(self.train_dataset, replacement=False),
        #     batch_size=self.args.per_device_train_batch_size,
        #     drop_last=True,
        # )

        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=effective_batch_size // self.num_generations,
            repeat_count=self.num_iterations,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    @profiling_decorator
    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }

            else:
                state_dict = unwrapped_model.state_dict()
                sampling = SamplingParams(max_tokens=10,
                                          temperature=0.0,
                                          skip_special_tokens=False)

            if self.accelerator.is_main_process:

                unwrapped_model.save_pretrained(  # ties are fixed & files are *.safetensors
                    self.model_dir,
                    safe_serialization=True,
                    # <-- avoids the shared-tensor crash:contentReference[oaicite:0]{index=0}
                )
                asyncio.run(self.vllm_client.reset_prefix_cache())
                asyncio.run(self.vllm_client.collective_rpc("load_model"))
                asyncio.run(self.vllm_client.reset_prefix_cache())
                torch.cuda.empty_cache()

                async def generate_once(prompt: str) -> str:
                    # vLLM returns an async generator: iterate until the final chunk
                    async for resp in self.vllm_client.generate(prompt, sampling, request_id=str(uuid.uuid4())):
                        pass  # consume the stream
                    return resp.outputs[0].text  # the full completion

                print(asyncio.run(generate_once("<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\nLet $x$ be a real number, $x > 1.$  Compute\n\\[\\sum_{n = 0}^\\infty \\frac{1}{x^{2^n} - x^{-2^n}}.\\]<|im_end|>\n<|im_start|>assistant\n")))


                # llm_model = self.vllm_client.llm_engine.model_executor.driver_worker.model_runner.model
                # llm_model = self.vllm_client.engine.model_executor.driver_worker.model_runner.model
                # llm_model.load_weights(state_dict.items())


                # asyncio.run(self.vllm_client.collective_rpc("load_model", state_dict, ))

            # Unmerge the adapter to restore the model to its original state.
            # This must be done after loading weights to ensure they correspond to the merged state.
            if is_peft_model(unwrapped_model):
                unwrapped_model.unmerge_adapter()

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        if self.control.should_save:
            try:
                self.log_results_200()
            except:
                pass
            torch.save(model.state_dict(), f"/workspace/{time.time()}_state_dict.pth")
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def training_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            num_items_in_batch=None,
    ) -> torch.Tensor:
        """GRPO optimisation step with **per‑prompt** loss averaging.

        Each *prompt* (root in the current batch) yields a list of sibling‑groups
        via :py:meth:`_prepare_inputs`.  We

        1. compute a loss for every sibling‑group;
        2. average those losses *within the same prompt*;
        3. perform **one** backward per prompt (lower variance, same gradient as
           all‑at‑once);
        4. return the mean of prompt losses for logging / scheduler.
        """

        import gc  # local import avoids top‑level pollution
        max_grad = getattr(self.args, "max_grad_norm", 1.0)  # clip value

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # ── Expand prompts into Tree‑of‑Thought group lists ──────────────────
        # ``batch_groups`` is List[List[group_dict]], len == batch size
        batch_groups: list[list[dict]] = self._prepare_inputs(inputs)

        total_prompt_losses: list[torch.Tensor] = []  # for scalar logging
        gc.collect()

        # ── Loop over prompts ────────────────────────────────────────────────
        for group_list in batch_groups:
            if not group_list:  # safety – can happen when a tree produced no groups
                continue

            prompt_losses: list[torch.Tensor] = []
            gc.collect()
            for group in group_list:
                with self.compute_loss_context_manager():
                    try:
                        loss = self._compute_loss_for_group(model, group)

                    except RuntimeError as err:
                        # OOM / NaNs – skip this group, free memory

                        print("[GRPO] Skipping group due to error:", err)
                        torch.cuda.empty_cache()
                        break

                if loss is not None and loss.requires_grad:
                    # scale so backward accumulates prompt‑mean
                    scaled_loss = loss / len(group_list)
                    try:
                        self.accelerator.backward(scaled_loss)
                    except:
                        print('OOM')
                        del group, loss
                        torch.cuda.empty_cache()
                        break


                    prompt_losses.append(loss.detach())  # for stats only

                # free ASAP
                del group, loss
                torch.cuda.empty_cache()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad, error_if_nonfinite=False)
            if not torch.isfinite(total_norm):
                if self.accelerator.is_main_process:
                    print(f"[GRPO] ⚠️  non‑finite grad (norm={total_norm}); "
                          "skipping this optimisation step.")
                self.optimizer.zero_grad(set_to_none=True)
                # return a dummy (detached) scalar so trainer keeps running
                return torch.zeros(1, device=self.accelerator.device)

            if prompt_losses:
                total_prompt_losses.append(torch.stack(prompt_losses).mean())

            del prompt_losses
            torch.cuda.empty_cache()


        if not total_prompt_losses:
            scalar_loss = torch.zeros(1, device=self.accelerator.device, requires_grad=False)
        else:
            scalar_loss = torch.stack(total_prompt_losses).mean()

        if self.args.n_gpu > 1:
            scalar_loss = scalar_loss.mean()

        if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        return scalar_loss.detach()

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"

        problems = [x["problem"] for x in inputs]
        final_answers= [x["final_answer"] for x in inputs]
        if mode == "train":
            buffer_index = self._step % self.args.gradient_accumulation_steps
            buffered_inputs = self._buffered_inputs[buffer_index]
            if self.state.global_step % self.num_iterations == 0 or buffered_inputs is None:
                # tree_root = run_async(self.tree_of_thought.expand_tree(problem, final_answer))
                trees = run_async(asyncio.gather(*[
                    self.tree_of_thought.expand_tree(p, a)
                    for p, a in zip(problems, final_answers)
                ]))
                inputs = [self._convert_tree_to_training_inputs(t) for t in trees]
                self._buffered_inputs[buffer_index] = inputs
            else:
                inputs = buffered_inputs
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            trees = run_async(asyncio.gather(*[
                self.tree_of_thought.expand_tree(p, a)
                for p, a in zip(problems, final_answers)
            ]))
            inputs = [self._convert_tree_to_training_inputs(t) for t in trees]
        return inputs

    def _convert_tree_to_training_inputs(self, tree_root) -> list[dict]:
        """Traverse *tree_root* and build group dictionaries.

        * If a node has **>2 children**, the resulting tensors can be large.
          To keep peak memory low we *chunk* such tensors into mini‑groups of
          at most **2 children each** before returning.  Down‑stream code
          (loss / backward) already loops over *every* group dict, so no other
          change is required.
        """


        if self.state.global_step != self._last_loaded_step:

            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        group_dicts: list[dict] = []  # output container
        pad_id = self.processing_class.pad_token_id

        # -------------------------------------------------- local helpers --
        def build_group(child_slice: slice, prom_ids, comp_ids, prom_mask,
                        comp_mask, adv, ref_lp):
            """Create a *single* group dict from slice indices."""
            return {
                "prompt_ids":           prom_ids[child_slice].clone(),
                "prompt_mask":          prom_mask[child_slice].clone(),
                "completion_ids":       comp_ids[child_slice].clone(),
                "completion_mask":      comp_mask[child_slice].clone(),
                "advantages":           adv[child_slice].clone(),
                "old_per_token_logps":  None,  # filled later if needed
                "ref_per_token_logps":  None if ref_lp is None else ref_lp[child_slice].clone(),
            }

        # -------------------------------------------------- DFS traversal --
        def process_node(node):
            if len(node.children) < 2:
                return  # need at least two siblings for a meaningful group

            child_nodes = [c for c in node.children if c.reward is not None]
            if len(child_nodes) < 2:
                return

            # ----- compute advantages ------------------------------------
            child_rewards = torch.tensor([c.reward for c in child_nodes], dtype=torch.float32)
            std_r = child_rewards.std()
            if std_r <= 1e-9:
                return  # degenerate reward distribution
            mean_r = child_rewards.mean()
            advantages = (child_rewards - mean_r)  # / std_r  (optional norm)

            # ----- build token tensors -----------------------------------
            prompt_ids      = torch.stack([torch.tensor(c.prompt_ids)     for c in child_nodes])
            completion_list = [torch.tensor(c.completion_ids)             for c in child_nodes]
            completion_ids  = pad(completion_list, padding_value=pad_id)

            prompt_mask     = torch.ones_like(prompt_ids)
            completion_mask = (completion_ids != pad_id).long()

            # ----- optional ref‑model log‑probs ---------------------------
            ref_per_token_lp = None
            if self.beta:
                with torch.no_grad():
                    pc_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                    att_m  = torch.cat([prompt_mask, completion_mask], dim=1)
                    logits_to_keep = completion_ids.size(1)

                    chunks = []
                    max_chunk = 2  # small chunk to save memory
                    for i in range(0, pc_ids.size(0), max_chunk):
                        p_chunk = pc_ids[i:i+max_chunk].to(self.ref_model.device)
                        m_chunk = att_m[i:i+max_chunk].to(self.ref_model.device)
                        out = self._get_per_token_logps(self.ref_model, p_chunk, m_chunk, logits_to_keep)
                        if out is None:
                            continue
                        chunks.append(out.cpu())
                    if chunks:
                        ref_per_token_lp = torch.cat(chunks, dim=0)

            # ----- split into smaller dicts if many children --------------
            num_children = prompt_ids.size(0)
            if num_children > 2:
                step = 2  # hard‑coded chunk of 2 rows
                for start in range(0, num_children, step):
                    sl = slice(start, min(start+step, num_children))
                    group_dicts.append(
                        build_group(sl, prompt_ids, completion_ids, prompt_mask,
                                     completion_mask, advantages, ref_per_token_lp)
                    )
            else:
                group_dicts.append(
                    build_group(slice(0, num_children), prompt_ids, completion_ids, prompt_mask,
                                 completion_mask, advantages, ref_per_token_lp)
                )

            # recurse into children
            for child in child_nodes:
                process_node(child)

        # kick‑off DFS
        process_node(tree_root)
        return group_dicts


    def _compute_loss_for_group(self, model, inputs):
        from collections import Counter
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        dtype_counts = Counter(param.dtype for param in model.parameters())
        try:
            chunk_threshold = 1200  # total elements threshold

            total_elements = input_ids.shape[0] * input_ids.shape[1]
            if total_elements > chunk_threshold:
                outputs = []
                # Process each row separately
                for i in range(input_ids.size(0)):
                    row_input_ids = input_ids[i:i + 1].to(model.device)
                    row_attention_mask = attention_mask[i:i + 1].to(model.device)
                    # Compute per-token log probabilities for this row
                    row_output = self._get_per_token_logps(model, row_input_ids, row_attention_mask, logits_to_keep)
                    outputs.append(row_output)
                    del row_output, row_attention_mask, row_input_ids
                    torch.cuda.empty_cache()
                per_token_logps = torch.cat(outputs, dim=0)#.to(model.device)
            else:
                per_token_logps = self._get_per_token_logps(model, input_ids.to(model.device),
                                                            attention_mask.to(model.device), logits_to_keep)
        except:

            for inpt in input_ids:
                print(self.tokenizer.decode(inpt))
            print(input_ids.shape)
            print(attention_mask)
            del input_ids, attention_mask
            torch.cuda.empty_cache()
            raise

        # Compute the KL divergence between the model and the reference model
        #del
        # torch.cuda.empty_cache()

        # Compute the loss
        advantages = inputs["advantages"].to(model.device)
        #Todo - insert adv normalizations
        # print('----- advantages ----')
        # print(advantages)
        # print('----- per_token_advantages ----')
        # token_lens = completion_mask.sum(dim=1).clamp(min=1).to(model.device)
        #
        # tok_adv = advantages.unsqueeze(1) / token_lens.unsqueeze(1)
        # print(tok_adv)
        # print('----- finished ----')

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        log_ratio = (per_token_logps - old_per_token_logps).clamp(-60, 60)
        coef_1 = torch.exp(log_ratio)
        # coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"].to(model.device)
            diff = (ref_per_token_logps - per_token_logps).clamp(-60, 60)
            per_token_kl = torch.exp(diff) - diff - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask.to(model.device)).sum() / completion_mask.sum().to(model.device)

        # for row_idx in range(prompt_ids.size(0)):
        #     print("▸ prompt", row_idx)
        #     print(self.tokenizer.decode(prompt_ids[row_idx]))
        #     print("▸ completion")
        #     print(self.tokenizer.decode(completion_ids[row_idx][completion_mask[row_idx].bool()]))
        #
        #     print("advantage:", float(advantages[row_idx]))
        #     print("per-token loss (first 5):", per_token_loss[row_idx][:5].tolist())
        # print("adv mean/var:", advantages.mean().item(), advantages.var(unbiased=False).item())
        # print("total loss:", loss.item(), "mean per-token:", per_token_loss.mean().item())


        del per_token_loss, per_token_loss2, per_token_loss1, completion_mask, advantages, coef_2, coef_1, per_token_logps
        torch.cuda.empty_cache()
        # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"

        # if self.beta != 0.0:
        #     mean_kl = (per_token_kl * completion_mask).sum() / MAX_TOKENS_TO_CALC_LOSS
        #     self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute the clip ratio
        # is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
        #         (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        # )
        # clip_ratio = (is_clipped * completion_mask).sum() / MAX_TOKENS_TO_CALC_LOSS
        # self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss



    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if isinstance(inputs, list):
            group_losses = [self._compute_loss_for_group(model, group) for group in inputs]
            if not group_losses:
                print("No valid groups for loss computation in this batch.")
                return torch.zeros(1, device=self.accelerator.device, requires_grad=True)
            loss = torch.stack(group_losses).mean()
            return loss
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute the clip ratio
        is_clipped = ((coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        model = model.to('cuda:0')
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))



if __name__ == '__main__':
    st = time.time()

    print()