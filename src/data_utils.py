"""
Data utility functions for downloading, preprocessing, formatting, and splitting 
the Chain-of-Thought (CoT) reasoning dataset for QLoRA SFT fine-tuning.
"""

import os
import json
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset, Dataset


def setup_environment(seed: int = 42) -> Dict[str, Any]:
    """
    Sets random seeds for reproducibility and prints system/GPU diagnostics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "allocated_vram_gb": round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0.0,
    }
    
    print(f"=== Environment Setup ===")
    print(f"CUDA Available: {env_info['cuda_available']}")
    if env_info['cuda_available']:
        print(f"GPU Device: {env_info['device_name']} (Count: {env_info['device_count']})")
    print(f"Random seed set to: {seed}")
    print("=========================\n")
    return env_info


def load_and_filter_dataset(
    dataset_name: str = "HuggingFaceH4/Bespoke-Stratos-17k",
    fallback_dataset_name: str = "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B",
    n_samples: int = 5000,
    seed: int = 42
) -> Dataset:
    """
    Loads dataset from HuggingFace, filters malformed/empty entries, and selects a subset.
    """
    print(f"Loading dataset: {dataset_name}...")
    try:
        raw_ds = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"Warning: Failed to load primary dataset ({e}). Loading fallback dataset: {fallback_dataset_name}")
        raw_ds = load_dataset(fallback_dataset_name, split="train")

    print(f"Original dataset size: {len(raw_ds)}")
    
    # Filter function depending on standard structures
    def is_valid(example):
        if "conversations" in example and example["conversations"]:
            return len(example["conversations"]) >= 2
        elif "messages" in example and example["messages"]:
            return len(example["messages"]) >= 2
        elif "problem" in example and "solution" in example:
            return bool(example["problem"]) and bool(example["solution"])
        elif "question" in example and "response" in example:
            return bool(example["question"]) and bool(example["response"])
        return True

    filtered_ds = raw_ds.filter(is_valid)
    print(f"Filtered dataset size: {len(filtered_ds)}")

    if len(filtered_ds) > n_samples:
        filtered_ds = filtered_ds.shuffle(seed=seed).select(range(n_samples))
        print(f"Subsampled to exactly {len(filtered_ds)} records.")

    return filtered_ds


def format_for_chat_template(sample: Dict[str, Any], model_type: str = "qwen") -> str:
    """
    Converts a single dataset sample into structured chat template format with explicit <think> tags.
    """
    system_prompt = "You are a helpful assistant that thinks step-by-step before answering."
    question = ""
    reasoning = ""
    answer = ""

    # Parse common fields across datasets
    if "conversations" in sample:
        convs = sample["conversations"]
        for turn in convs:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ["user", "human"]:
                question = content
            elif role in ["assistant", "gpt"]:
                answer = content
    elif "messages" in sample:
        msgs = sample["messages"]
        for turn in msgs:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                question = content
            elif role == "assistant":
                answer = content
    elif "problem" in sample and "solution" in sample:
        question = sample["problem"]
        answer = sample["solution"]
    elif "question" in sample and "response" in sample:
        question = sample["question"]
        answer = sample["response"]

    # Extract intermediate reasoning trace if separate tags exist
    if "<think>" in answer and "</think>" in answer:
        parts = answer.split("</think>")
        reasoning = parts[0].replace("<think>", "").strip()
        answer = parts[1].strip()
    elif "system" in sample and "thought" in sample:
        reasoning = sample["thought"]

    # Format output according to standard ChatML format
    if model_type.lower() == "qwen":
        formatted_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        if reasoning:
            formatted_text += f"<think>\n{reasoning}\n</think>\n"
        formatted_text += f"{answer}<|im_end|>"
    else:  # General Llama style
        formatted_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        if reasoning:
            formatted_text += f"<think>\n{reasoning}\n</think>\n"
        formatted_text += f"{answer}<|eot_id|>"

    return formatted_text


def prepare_splits(dataset: Dataset, train_ratio: float = 0.9, seed: int = 42, model_type: str = "qwen") -> Tuple[Dataset, Dataset]:
    """
    Formats the dataset and splits it into training and testing sets.
    """
    formatted_data = []
    for item in dataset:
        text = format_for_chat_template(item, model_type=model_type)
        formatted_data.append({"text": text})

    formatted_ds = Dataset.from_list(formatted_data)
    split_ds = formatted_ds.train_test_split(train_size=train_ratio, seed=seed)
    
    print(f"Train split size: {len(split_ds['train'])}")
    print(f"Test split size: {len(split_ds['test'])}")
    return split_ds["train"], split_ds["test"]


def compute_token_stats(dataset: Dataset, tokenizer: Any) -> Dict[str, Any]:
    """
    Computes token length distribution statistics for the dataset.
    """
    token_lengths = [len(tokenizer.encode(item["text"])) for item in dataset]
    
    stats = {
        "count": len(token_lengths),
        "mean": float(np.mean(token_lengths)),
        "median": float(np.median(token_lengths)),
        "max": int(np.max(token_lengths)),
        "p95": float(np.percentile(token_lengths, 95)),
        "p99": float(np.percentile(token_lengths, 99)),
    }
    
    print("=== Token Length Statistics ===")
    print(f"Mean Length: {stats['mean']:.2f}")
    print(f"Median Length: {stats['median']:.2f}")
    print(f"P95 Length: {stats['p95']:.2f}")
    print(f"P99 Length: {stats['p99']:.2f}")
    print(f"Max Length: {stats['max']}")
    print("===============================\n")
    return stats


def save_splits(train_ds: Dataset, test_ds: Dataset, output_dir: str = "data") -> None:
    """
    Saves dataset splits locally into jsonl format.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    
    train_ds.to_json(train_path, orient="records", lines=True)
    test_ds.to_json(test_path, orient="records", lines=True)
    
    print(f"Successfully saved splits to '{train_path}' and '{test_path}'.")
