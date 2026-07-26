import os
import sys

# Ensure current script directory and project paths are in sys.path BEFORE importing from src
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
kaggle_path = "/kaggle/working/SLMs"

for p in [script_dir, cwd, kaggle_path]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import yaml
import json
import torch
from datasets import Dataset

from src.data_utils import setup_environment
from src.model_utils import load_base_model, apply_qlora, get_memory_stats, clear_gpu_memory
from src.training_config import get_training_args, VRAMLoggingCallback

try:
    from trl import SFTTrainer
except ImportError:
    from transformers import Trainer as SFTTrainer


def main():
    print("=== Executing Phase 3: QLoRA Fine-Tuning for CoT Reasoning ===")

    # 1. Load project configuration
    config_path = os.path.join(script_dir, "configs", "qlora_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    setup_environment(seed=config['project']['seed'])

    # 2. Load Base Model & Tokenizer using Unsloth 4-bit
    model, tokenizer = load_base_model(
        model_name=config['model']['base_model_name'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit']
    )

    # 3. Apply QLoRA PEFT Adapters (r=16, alpha=32)
    model = apply_qlora(
        model=model,
        r=config['qlora']['r'],
        lora_alpha=config['qlora']['lora_alpha'],
        lora_dropout=config['qlora']['lora_dropout'],
        target_modules=config['qlora']['target_modules']
    )

    # 4. Load Processed Dataset Splits from Phase 1
    train_file = os.path.join(script_dir, config['data']['train_file'])
    test_file = os.path.join(script_dir, config['data']['test_file'])

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training dataset '{train_file}' not found. Please run Phase 1 first.")

    print(f"Loading dataset splits from '{train_file}' and '{test_file}'...")
    train_dataset = Dataset.from_json(train_file)
    test_dataset = Dataset.from_json(test_file)
    print(f"Loaded Train Samples: {len(train_dataset)}, Test Samples: {len(test_dataset)}")

    # 5. Set up Training Arguments & Callbacks
    output_model_dir = os.path.join(script_dir, config['training']['output_dir'])
    training_args = get_training_args(
        output_dir=output_model_dir,
        num_epochs=config['training']['num_train_epochs'],
        batch_size=config['training']['per_device_train_batch_size'],
        grad_accum=config['training']['gradient_accumulation_steps'],
        lr=config['training']['learning_rate'],
        max_seq_length=config['model']['max_seq_length'],
        seed=config['project']['seed']
    )

    vram_callback = VRAMLoggingCallback()

    # 6. Initialize SFTTrainer
    print("Initializing SFTTrainer with fast Unsloth integration...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        args=training_args,
        callbacks=[vram_callback],
    )

    # 7. Execute Fine-Tuning
    print("\n🚀 Starting QLoRA Fine-Tuning (3 Epochs)...")
    train_result = trainer.train()

    # 8. Save Trained LoRA Adapter Weights
    print("\n✅ Training finished successfully! Saving LoRA adapter weights...")
    os.makedirs(output_model_dir, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Saved LoRA adapters to '{output_model_dir}'.")

    # 9. Save Training Metrics & VRAM Profile
    results_dir = os.path.join(script_dir, config['evaluation']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    logs_filepath = os.path.join(results_dir, "training_logs.json")

    log_data = {
        "train_runtime_sec": train_result.metrics.get("train_runtime", 0.0),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0.0),
        "train_loss": train_result.metrics.get("train_loss", 0.0),
        "vram_logs": vram_callback.vram_logs,
        "peak_vram_gb": get_memory_stats().get("max_allocated_gb", 0.0)
    }

    with open(logs_filepath, "w") as f:
        json.dump(log_data, f, indent=2)

    print(f"Saved training log traces to '{logs_filepath}'.")
    print(f"Peak VRAM Allocated: {log_data['peak_vram_gb']} GB")
    print("==========================================")
    print("=== Phase 3 Completed Successfully ===")
    print("==========================================")

    clear_gpu_memory()

if __name__ == "__main__":
    main()
