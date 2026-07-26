"""
Evaluation utilities for CoT reasoning metrics, answer extraction, GSM8K, and ARC benchmarks.
Optimized for ultra-fast execution on Kaggle Dual T4 GPUs.
"""

import re
import json
import os
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from rouge_score import rouge_scorer
from datasets import load_dataset

from src.prompts import format_cot_prompt, format_gsm8k_prompt, format_arc_prompt


def extract_thinking(response_text: str) -> str:
    """Extracts intermediate reasoning trace (<think>...</think>) from response."""
    if "<think>" in response_text and "</think>" in response_text:
        match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def extract_final_answer(response_text: str) -> str:
    """Extracts final answer from model response."""
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    pattern_match = re.search(r"final answer is[:\s]*([^\.\n]+)", response_text, re.IGNORECASE)
    if pattern_match:
        return pattern_match.group(1).strip()

    lines = [l.strip() for l in response_text.split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return response_text.strip()


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """Computes Exact Match (EM) accuracy score."""
    matches = 0
    for pred, ref in zip(predictions, references):
        norm_pred = re.sub(r"[^\w\s]", "", pred.lower()).strip()
        norm_ref = re.sub(r"[^\w\s]", "", ref.lower()).strip()
        if norm_pred == norm_ref or norm_ref in norm_pred:
            matches += 1
    return round((matches / len(predictions)) * 100.0, 2) if predictions else 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """Computes average ROUGE-L F1 score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        if not ref.strip():
            continue
        score = scorer.score(ref, pred)['rougeL'].fmeasure
        scores.append(score)
    return round((sum(scores) / len(scores)) * 100.0, 2) if scores else 0.0


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    batch_size: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 0.0
) -> List[str]:
    """
    Ultra-fast batched generation using Unsloth native inference optimization.
    """
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    responses = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    print(f"Generating responses for {len(prompts)} items in {total_batches} GPU batches (batch_size={batch_size})...")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1e-5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
        for idx, output in enumerate(outputs):
            input_len = inputs.input_ids[idx].shape[0]
            generated_ids = output[input_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response)

        current_count = min(i + batch_size, len(prompts))
        print(f"  --> Processed {current_count}/{len(prompts)} prompts...")

    return responses


def evaluate_cot_testset(
    model: Any,
    tokenizer: Any,
    test_jsonl_path: str,
    n_samples: int = 100,
    model_type: str = "qwen"
) -> Dict[str, Any]:
    """Evaluates model on held-out CoT test set."""
    print(f"\n[1/3] Evaluating CoT Test Set ({n_samples} samples)...")
    if not os.path.exists(test_jsonl_path):
        print(f"Warning: Test file {test_jsonl_path} not found.")
        return {"exact_match": 0.0, "rouge_l": 0.0, "samples": []}

    samples = []
    with open(test_jsonl_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= n_samples:
                break

    prompts = []
    references = []
    ref_answers = []

    for s in samples:
        text = s["text"]
        if "<|im_start|>assistant" in text:
            parts = text.split("<|im_start|>assistant\n")
            prompt = parts[0] + "<|im_start|>assistant\n"
            ref = parts[1].replace("<|im_end|>", "").strip()
        else:
            prompt = text[:len(text)//2]
            ref = text[len(text)//2:]

        prompts.append(prompt)
        references.append(ref)
        ref_answers.append(extract_final_answer(ref))

    responses = batch_generate(model, tokenizer, prompts, batch_size=16, max_new_tokens=256)
    pred_answers = [extract_final_answer(r) for r in responses]

    em = compute_exact_match(pred_answers, ref_answers)
    rouge_l = compute_rouge_l(responses, references)

    print(f"✅ CoT Test Set Results -> EM: {em}%, ROUGE-L: {rouge_l}%")

    qualitative_samples = []
    for i in range(min(5, len(prompts))):
        qualitative_samples.append({
            "prompt": prompts[i][:200] + "...",
            "reference": references[i][:300],
            "generated": responses[i][:300],
            "extracted_pred": pred_answers[i],
            "extracted_ref": ref_answers[i]
        })

    return {
        "exact_match": em,
        "rouge_l": rouge_l,
        "n_samples": len(samples),
        "qualitative_examples": qualitative_samples
    }


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    n_samples: int = 100
) -> Dict[str, Any]:
    """Evaluates zero-shot math reasoning on GSM8K benchmark."""
    print(f"\n[2/3] Evaluating GSM8K ({n_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    prompts = [format_gsm8k_prompt(item["question"]) for item in ds]
    ref_answers = [item["answer"].split("####")[-1].strip() for item in ds]

    responses = batch_generate(model, tokenizer, prompts, batch_size=16, max_new_tokens=256)
    pred_answers = [extract_final_answer(r) for r in responses]

    acc = compute_exact_match(pred_answers, ref_answers)
    print(f"✅ GSM8K Accuracy: {acc}%")

    return {
        "gsm8k_accuracy": acc,
        "n_samples": len(ds)
    }


def evaluate_arc_challenge(
    model: Any,
    tokenizer: Any,
    n_samples: int = 100
) -> Dict[str, Any]:
    """Evaluates zero-shot science reasoning on ARC-Challenge benchmark."""
    print(f"\n[3/3] Evaluating ARC-Challenge ({n_samples} samples)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    prompts = [format_arc_prompt(item["question"], item["choices"]) for item in ds]
    ref_answers = [item["answerKey"] for item in ds]

    responses = batch_generate(model, tokenizer, prompts, batch_size=16, max_new_tokens=128)
    pred_answers = [extract_final_answer(r) for r in responses]

    acc = compute_exact_match(pred_answers, ref_answers)
    print(f"✅ ARC-Challenge Accuracy: {acc}%")

    return {
        "arc_accuracy": acc,
        "n_samples": len(ds)
    }


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Saves benchmark results into structured JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to '{filepath}'.")
