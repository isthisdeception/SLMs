"""
Evaluation utilities for CoT reasoning metrics, answer extraction, GSM8K, and ARC benchmarks.
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
    """
    Extracts the intermediate reasoning trace (<think>...</think>) from a model response.
    """
    if "<think>" in response_text and "</think>" in response_text:
        match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""


def extract_final_answer(response_text: str) -> str:
    """
    Extracts the final answer from a model response (boxed, pattern matching, or last line).
    """
    # Clean output by removing reasoning block if present
    if "</think>" in response_text:
        response_text = response_text.split("</think>")[-1].strip()

    # Case 1: \\boxed{answer} format
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Case 2: "The final answer is [X]" or "The answer is [X]"
    pattern_match = re.search(r"final answer is[:\s]*([^\.\n]+)", response_text, re.IGNORECASE)
    if pattern_match:
        return pattern_match.group(1).strip()

    # Case 3: Fallback to last non-empty line
    lines = [l.strip() for l in response_text.split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return response_text.strip()


def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Computes Exact Match (EM) accuracy between normalized prediction strings and reference answers.
    """
    matches = 0
    for pred, ref in zip(predictions, references):
        norm_pred = re.sub(r"[^\w\s]", "", pred.lower()).strip()
        norm_ref = re.sub(r"[^\w\s]", "", ref.lower()).strip()
        if norm_pred == norm_ref or norm_ref in norm_pred:
            matches += 1
    return round((matches / len(predictions)) * 100.0, 2) if predictions else 0.0


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    Computes average ROUGE-L F1 score for generated text against reference text.
    """
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
    max_new_tokens: int = 512,
    temperature: float = 0.0
) -> List[str]:
    """
    Generates model responses for a list of input prompts using greedy decoding.
    """
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
    except ImportError:
        model.eval()

    responses = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for prompt in tqdm(prompts, desc="Generating responses"):
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1e-5,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response)

    return responses


def evaluate_cot_testset(
    model: Any,
    tokenizer: Any,
    test_jsonl_path: str,
    n_samples: int = 200,
    model_type: str = "qwen"
) -> Dict[str, Any]:
    """
    Evaluates model on held-out CoT test set for EM and ROUGE-L metrics.
    """
    print(f"Evaluating CoT Test Set ({n_samples} samples)...")
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
        # Split text into prompt (up to assistant header) and reference target
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

    responses = batch_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.0)
    pred_answers = [extract_final_answer(r) for r in responses]

    em = compute_exact_match(pred_answers, ref_answers)
    rouge_l = compute_rouge_l(responses, references)

    print(f"CoT Test Set Results -> EM: {em}%, ROUGE-L: {rouge_l}%")

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
    n_samples: int = 200
) -> Dict[str, Any]:
    """
    Evaluates zero-shot math reasoning on GSM8K benchmark.
    """
    print(f"Evaluating GSM8K ({n_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    prompts = [format_gsm8k_prompt(item["question"]) for item in ds]
    ref_answers = [item["answer"].split("####")[-1].strip() for item in ds]

    responses = batch_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.0)
    pred_answers = [extract_final_answer(r) for r in responses]

    acc = compute_exact_match(pred_answers, ref_answers)
    print(f"GSM8K Accuracy: {acc}%")

    return {
        "gsm8k_accuracy": acc,
        "n_samples": len(ds)
    }


def evaluate_arc_challenge(
    model: Any,
    tokenizer: Any,
    n_samples: int = 200
) -> Dict[str, Any]:
    """
    Evaluates zero-shot science reasoning on ARC-Challenge benchmark.
    """
    print(f"Evaluating ARC-Challenge ({n_samples} samples)...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    prompts = [format_arc_prompt(item["question"], item["choices"]) for item in ds]
    ref_answers = [item["answerKey"] for item in ds]

    responses = batch_generate(model, tokenizer, prompts, max_new_tokens=256, temperature=0.0)
    pred_answers = [extract_final_answer(r) for r in responses]

    acc = compute_exact_match(pred_answers, ref_answers)
    print(f"ARC-Challenge Accuracy: {acc}%")

    return {
        "arc_accuracy": acc,
        "n_samples": len(ds)
    }


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Saves evaluation benchmark results dictionary into structured JSON.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to '{filepath}'.")
