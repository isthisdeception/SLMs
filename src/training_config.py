"""
Training configuration, custom callbacks, and SFTTrainer setup for QLoRA fine-tuning.
"""

import os
import torch
import yaml
from typing import Dict, Any, Tuple, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    from trl import SFTTrainer
    SFTConfig = None

from transformers import TrainingArguments


class VRAMLoggingCallback(TrainerCallback):
    """
    Custom Trainer Callback to log GPU memory utilization at every logging step.
    """
    def __init__(self):
        self.vram_logs = []

    def on_log(self, args: Any, state: TrainerState, control: TrainerControl, logs: Optional[Dict[str, float]] = None, **kwargs):
        if torch.cuda.is_available():
            allocated_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            max_allocated_gb = round(torch.cuda.max_memory_allocated(0) / (1024**3), 2)
            step_info = {
                "step": state.global_step,
                "epoch": state.epoch,
                "allocated_gb": allocated_gb,
                "max_allocated_gb": max_allocated_gb
            }
            if logs:
                step_info.update(logs)
            self.vram_logs.append(step_info)


def get_training_args(
    output_dir: str = "models/qwen2.5-3b-cot-qlora",
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_seq_length: int = 2048,
    seed: int = 42
) -> Any:
    """
    Constructs optimized TrainingArguments / SFTConfig for Kaggle T4/P100 GPUs.
    """
    if SFTConfig is not None:
        return SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            packing=True,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            report_to="none",
            optim="adamw_8bit"
        )
    else:
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            fp16=True,
            bf16=False,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            seed=seed,
            report_to="none",
            optim="adamw_8bit"
        )
