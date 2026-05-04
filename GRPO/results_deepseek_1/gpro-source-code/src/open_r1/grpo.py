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

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training

# Monkey patches for third-party libraries (like llm-blender, etc.) which rely on deprecated transformers variables
import transformers.utils.hub
import transformers.utils

# 1. Caching paths
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = os.getenv("HF_HOME", "/root/.cache/huggingface/hub")
if not hasattr(transformers.utils.hub, "default_cache_path"):
    transformers.utils.hub.default_cache_path = os.getenv("HF_HOME", "/root/.cache/huggingface/hub")

# 2. Legacy environment variable sets
if not hasattr(transformers.utils, "ENV_VARS_TRUE_VALUES"):
    transformers.utils.ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
if not hasattr(transformers.utils, "ENV_VARS_TRUE_AND_AUTO_VALUES"):
    transformers.utils.ENV_VARS_TRUE_AND_AUTO_VALUES = {"1", "ON", "YES", "TRUE", "AUTO"}

# 3. Legacy file_utils mapping
if not hasattr(transformers, "file_utils"):
    transformers.file_utils = transformers.utils

# 4. Strip use_cache from from_pretrained kwargs because TRL aggressively injects it
from transformers import AutoModelForCausalLM
_original_from_pretrained = AutoModelForCausalLM.from_pretrained

@classmethod
def _patched_from_pretrained(cls, *args, **kwargs):
    kwargs.pop("use_cache", None)
    model = _original_from_pretrained(*args, **kwargs)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model

AutoModelForCausalLM.from_pretrained = _patched_from_pretrained

from trl import GRPOTrainer as _OriginalGRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

# 5. Subclass GRPOTrainer to fix signature mismatch:
#    New transformers calls _get_train_sampler(dataset) but old TRL has (self) only.
class GRPOTrainer(_OriginalGRPOTrainer):
    def _get_train_sampler(self, dataset=None):
        return super()._get_train_sampler()

# 6. Patch Qwen tokenizer for vLLM compatibility:
#    vLLM 0.7.2 calls tokenizer.all_special_tokens_extended which was removed
#    from newer transformers tokenizer classes.
from transformers import Qwen2Tokenizer
if not hasattr(Qwen2Tokenizer, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return list(self.all_special_tokens)
    Qwen2Tokenizer.all_special_tokens_extended = _all_special_tokens_extended

# Also patch the base class in case vLLM uses a different tokenizer class
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _base_all_special_tokens_extended(self):
        return list(self.all_special_tokens)
    PreTrainedTokenizerBase.all_special_tokens_extended = _base_all_special_tokens_extended

# 7. Force vLLM to use float16 + TORCH_SDPA backend (avoids XFormers device mismatch on T4).
#    When vllm_device=cuda:1 but default device=cuda:0, XFormers creates attn_bias on wrong device.
#    TORCH_SDPA uses pure PyTorch ops that respect the model device correctly.
import os as _os
_os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"

try:
    import vllm as _vllm_module
    _original_LLM_init = _vllm_module.LLM.__init__
    def _patched_LLM_init(self, *args, **kwargs):
        import torch as _torch
        # Set default device to cuda:1 so all vLLM internal tensors land correctly
        _vllm_device = kwargs.get("device", "cuda:0")
        _vllm_gpu_id = int(str(_vllm_device).split(":")[-1]) if ":" in str(_vllm_device) else 0
        _prev_device = _torch.cuda.current_device()
        _torch.cuda.set_device(_vllm_gpu_id)
        kwargs["dtype"] = "float16"
        try:
            _original_LLM_init(self, *args, **kwargs)
        finally:
            _torch.cuda.set_device(_prev_device)  # Always restore original device
    _vllm_module.LLM.__init__ = _patched_LLM_init

    # Also patch ModelConfig to override dtype at the engine level
    from vllm.config import ModelConfig as _VLLMModelConfig
    _original_model_config_init = _VLLMModelConfig.__init__
    def _patched_model_config_init(self, *args, **kwargs):
        kwargs["dtype"] = "float16"
        _original_model_config_init(self, *args, **kwargs)
    _VLLMModelConfig.__init__ = _patched_model_config_init
except Exception:
    pass  # vLLM not installed or API changed, skip

# 8. Fix XFormers device mismatch when vLLM runs on non-default GPU (cuda:1).
#    XFormers creates attn_bias on cuda:0 (default), but vLLM model is on cuda:1.
#    We patch Inputs.validate_inputs to auto-move attn_bias to the query device.
try:
    import xformers.ops.fmha.common as _xf_common
    _original_validate_inputs = _xf_common.Inputs.validate_inputs
    def _patched_validate_inputs(self):
        if self.attn_bias is not None and hasattr(self.attn_bias, "device"):
            q_device = self.query.device
            if self.attn_bias.device != q_device:
                self.attn_bias = self.attn_bias.to(q_device)
        _original_validate_inputs(self)
    _xf_common.Inputs.validate_inputs = _patched_validate_inputs
except Exception:
    pass  # xformers not installed or API changed

logger = logging.getLogger(__name__)


class StepMetricsCallback(TrainerCallback):
    """Lưu chi tiết metrics từng step: loss, reward, GPU memory, thời gian."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.rows = []
        self._step_start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        import time
        self._step_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time, torch
        if logs is None or state.global_step == 0:
            return
        # Chỉ lưu trên main process
        if not state.is_world_process_zero:
            return

        step_time = (time.time() - self._step_start_time) if self._step_start_time else 0.0

        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4)}

        # Loss, LR, grad_norm
        for key in ("loss", "learning_rate", "grad_norm"):
            row[key] = logs.get(key, None)

        # Tất cả reward metrics (format_reward, cosine_reward, ...)
        for key, val in logs.items():
            if "reward" in key:
                row[key] = val

        # GPU memory (chỉ GPU 0 — rank 0)
        if torch.cuda.is_available():
            row["gpu_mem_alloc_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            row["gpu_mem_peak_mb"]  = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        else:
            row["gpu_mem_alloc_mb"] = row["gpu_mem_peak_mb"] = 0.0

        row["step_time_sec"] = round(step_time, 2)
        self.rows.append(row)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.rows:
            return
        import csv, os
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "step_metrics.csv")
        fieldnames = list(self.rows[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        logger.info(f"✅ Saved per-step metrics ({len(self.rows)} steps) → {out_path}")

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args) + [StepMetricsCallback(training_args.output_dir)],
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    import time
    def get_model_size(model):
        return sum(p.numel() * p.element_size() for p in model.parameters())

    total_size_before = get_model_size(trainer.model)
    start_time = time.time()

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Custom Metrics Tracking
    total_time = time.time() - start_time
    total_size_after = get_model_size(trainer.model)
    
    if torch.cuda.is_available():
        ram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        ram_consump = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        ram_peak, ram_consump = 0.0, 0.0
        
    def get_dir_size(path="."):
        total = 0
        if not os.path.exists(path): return 0
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_dir_size(entry.path)
        return total
        
    disk_storage = get_dir_size(training_args.output_dir) / (1024 ** 2)

    if trainer.accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
        metrics_file = os.path.join(training_args.output_dir, "training_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"total_size_before (MB): {total_size_before / (1024**2):.2f}\n")
            f.write(f"total_size_after (MB): {total_size_after / (1024**2):.2f}\n")
            f.write(f"total_time (seconds): {total_time:.2f}\n")
            f.write(f"ram_peak (MB): {ram_peak:.2f}\n")
            f.write(f"ram_consump (MB): {ram_consump:.2f}\n")
            f.write(f"disk_storage (MB): {disk_storage:.2f}\n")
        logger.info(f"Saved custom training metrics to {metrics_file}")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
