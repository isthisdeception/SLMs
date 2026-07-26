import os
import sys
import yaml
import json

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import setup_environment
from src.model_utils import load_base_model, get_memory_stats, clear_gpu_memory
from src.eval_utils import evaluate_cot_testset, evaluate_gsm8k, evaluate_arc_challenge, save_results

def main():
    print("=== Executing Phase 2: Model Setup & Baseline Evaluation ===")
    
    # 1. Load config
    config_path = os.path.join(PROJECT_ROOT, "configs", "qlora_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    env_info = setup_environment(seed=config['project']['seed'])

    # 2. Load Base Model (Zero-Shot)
    model, tokenizer = load_base_model(
        model_name=config['model']['base_model_name'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit']
    )

    initial_memory = get_memory_stats()
    print(f"Base Model Memory Profile: {initial_memory}")

    # 3. Evaluate on CoT held-out test set
    test_jsonl_path = os.path.join(PROJECT_ROOT, config['data']['test_file'])
    cot_results = evaluate_cot_testset(
        model=model,
        tokenizer=tokenizer,
        test_jsonl_path=test_jsonl_path,
        n_samples=config['evaluation']['cot_test_samples'],
        model_type=config['model']['model_type']
    )

    # 4. Evaluate on GSM8K benchmark
    gsm8k_results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        n_samples=config['evaluation']['gsm8k_samples']
    )

    # 5. Evaluate on ARC-Challenge benchmark
    arc_results = evaluate_arc_challenge(
        model=model,
        tokenizer=tokenizer,
        n_samples=config['evaluation']['arc_samples']
    )

    # 6. Aggregate baseline results
    baseline_summary = {
        "model_name": config['model']['base_model_name'],
        "memory_profile": initial_memory,
        "cot_exact_match": cot_results.get("exact_match", 0.0),
        "cot_rouge_l": cot_results.get("rouge_l", 0.0),
        "gsm8k_accuracy": gsm8k_results.get("gsm8k_accuracy", 0.0),
        "arc_accuracy": arc_results.get("arc_accuracy", 0.0),
        "qualitative_examples": cot_results.get("qualitative_examples", [])
    }

    # 7. Save results
    results_filepath = os.path.join(PROJECT_ROOT, config['evaluation']['results_dir'], "baseline_results.json")
    save_results(baseline_summary, results_filepath)

    print("\n==========================================")
    print("      BASELINE EVALUATION SUMMARY         ")
    print("==========================================")
    print(f"Model: {baseline_summary['model_name']}")
    print(f"CoT Test Set Exact Match (EM): {baseline_summary['cot_exact_match']}%")
    print(f"CoT Test Set ROUGE-L: {baseline_summary['cot_rouge_l']}%")
    print(f"GSM8K Accuracy: {baseline_summary['gsm8k_accuracy']}%")
    print(f"ARC-Challenge Accuracy: {baseline_summary['arc_accuracy']}%")
    print("==========================================")
    print("=== Phase 2 Completed Successfully ===")

    clear_gpu_memory()

if __name__ == "__main__":
    main()
