import os
import sys

# Ensure current script directory and project paths are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()
kaggle_path = "/kaggle/working/SLMs"

for p in [script_dir, cwd, kaggle_path]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from peft import PeftModel

from src.data_utils import setup_environment
from src.model_utils import load_base_model, get_memory_stats, clear_gpu_memory
from src.eval_utils import evaluate_cot_testset, evaluate_gsm8k, evaluate_arc_challenge, save_results

def generate_plots_and_latex(baseline: dict, sft: dict, logs: dict, output_dir: str):
    """
    Generates publication-quality figures and LaTeX table snippets for the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Bar Chart: Accuracy Benchmark Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    benchmarks = ["CoT Test (EM)", "GSM8K (Math)", "ARC-Challenge"]
    baseline_scores = [baseline.get("cot_exact_match", 0), baseline.get("gsm8k_accuracy", 0), baseline.get("arc_accuracy", 0)]
    sft_scores = [sft.get("cot_exact_match", 0), sft.get("gsm8k_accuracy", 0), sft.get("arc_accuracy", 0)]
    
    x = range(len(benchmarks))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], baseline_scores, width, label='Baseline (Zero-Shot)', color='#7f7f7f', edgecolor='black')
    ax.bar([i + width/2 for i in x], sft_scores, width, label='QLoRA SFT (Ours)', color='#1f77b4', edgecolor='black')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Reasoning Performance: Baseline vs. QLoRA SFT', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    
    for i in x:
        ax.text(i - width/2, baseline_scores[i] + 1.5, f"{baseline_scores[i]:.1f}%", ha='center', fontsize=10)
        ax.text(i + width/2, sft_scores[i] + 1.5, f"{sft_scores[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved figure to '{chart_path}'.")

    # 2. LaTeX Table Generation
    latex_table = r"""\begin{table}[h]
\centering
\caption{\textbf{Main Results:} Comparison of Base Model (Zero-Shot) vs. QLoRA Fine-Tuned Model across Reasoning Benchmarks.}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{CoT EM (\%)} & \textbf{CoT ROUGE-L} & \textbf{GSM8K (\%)} & \textbf{ARC-C (\%)} \\
\midrule
"""
    latex_table += f"Qwen2.5-3B (Baseline) & {baseline_scores[0]:.1f} & {baseline.get('cot_rouge_l', 0):.1f} & {baseline_scores[1]:.1f} & {baseline_scores[2]:.1f} \\\\\n"
    latex_table += f"Qwen2.5-3B (QLoRA SFT) & \\textbf{{{sft_scores[0]:.1f}}} & \\textbf{{{sft.get('cot_rouge_l', 0):.1f}}} & \\textbf{{{sft_scores[1]:.1f}}} & \\textbf{{{sft_scores[2]:.1f}}} \\\\\n"
    delta_em = sft_scores[0] - baseline_scores[0]
    delta_gsm = sft_scores[1] - baseline_scores[1]
    delta_arc = sft_scores[2] - baseline_scores[2]
    latex_table += f"\\midrule\n\\textbf{{Absolute Improvement ($\\Delta$)}} & \\textbf{{+{delta_em:.1f}\\%}} & -- & \\textbf{{+{delta_gsm:.1f}\\%}} & \\textbf{{+{delta_arc:.1f}\\%}} \\\\\n"
    latex_table += r"""\bottomrule
\end{tabular}
\end{table}
"""

    latex_path = os.path.join(output_dir, "main_results_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to '{latex_path}'.")


def main():
    print("=== Executing Phase 4: Post-Training Evaluation & Analysis ===")

    # 1. Load configuration
    config_path = os.path.join(script_dir, "configs", "qlora_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    setup_environment(seed=config['project']['seed'])

    # 2. Load Base Model
    model, tokenizer = load_base_model(
        model_name=config['model']['base_model_name'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit']
    )

    # 3. Load Trained QLoRA Adapters from Phase 3
    adapter_path = os.path.join(script_dir, config['training']['output_dir'])
    if os.path.exists(adapter_path):
        print(f"Loading trained QLoRA adapters from '{adapter_path}'...")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"Warning: Adapter path '{adapter_path}' not found. Evaluating base model...")

    # 4. Evaluate Fine-Tuned Model on CoT Test Set
    test_jsonl_path = os.path.join(script_dir, config['data']['test_file'])
    cot_results = evaluate_cot_testset(
        model=model,
        tokenizer=tokenizer,
        test_jsonl_path=test_jsonl_path,
        n_samples=config['evaluation']['cot_test_samples'],
        model_type=config['model']['model_type']
    )

    # 5. Evaluate on GSM8K
    gsm8k_results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        n_samples=config['evaluation']['gsm8k_samples']
    )

    # 6. Evaluate on ARC-Challenge
    arc_results = evaluate_arc_challenge(
        model=model,
        tokenizer=tokenizer,
        n_samples=config['evaluation']['arc_samples']
    )

    # 7. Save SFT Results
    sft_summary = {
        "model_name": f"{config['model']['base_model_name']}-QLoRA-SFT",
        "cot_exact_match": cot_results.get("exact_match", 0.0),
        "cot_rouge_l": cot_results.get("rouge_l", 0.0),
        "gsm8k_accuracy": gsm8k_results.get("gsm8k_accuracy", 0.0),
        "arc_accuracy": arc_results.get("arc_accuracy", 0.0),
        "qualitative_examples": cot_results.get("qualitative_examples", [])
    }

    results_dir = os.path.join(script_dir, config['evaluation']['results_dir'])
    sft_filepath = os.path.join(results_dir, "sft_results.json")
    save_results(sft_summary, sft_filepath)

    # 8. Compare with Baseline and Generate Figures
    baseline_filepath = os.path.join(results_dir, "baseline_results.json")
    baseline_summary = {}
    if os.path.exists(baseline_filepath):
        with open(baseline_filepath, "r") as f:
            baseline_summary = json.load(f)

    training_logs_filepath = os.path.join(results_dir, "training_logs.json")
    training_logs = {}
    if os.path.exists(training_logs_filepath):
        with open(training_logs_filepath, "r") as f:
            training_logs = json.load(f)

    figures_dir = os.path.join(script_dir, "paper", "figures")
    generate_plots_and_latex(baseline_summary, sft_summary, training_logs, figures_dir)

    print("\n==========================================")
    print("      POST-TRAINING EVALUATION SUMMARY    ")
    print("==========================================")
    print(f"Model: {sft_summary['model_name']}")
    print(f"CoT Test Exact Match:  Baseline {baseline_summary.get('cot_exact_match', 0)}% -> SFT {sft_summary['cot_exact_match']}%")
    print(f"GSM8K Accuracy:        Baseline {baseline_summary.get('gsm8k_accuracy', 0)}% -> SFT {sft_summary['gsm8k_accuracy']}%")
    print(f"ARC-Challenge Accuracy: Baseline {baseline_summary.get('arc_accuracy', 0)}% -> SFT {sft_summary['arc_accuracy']}%")
    print("==========================================")
    print("=== Phase 4 Completed Successfully ===")

    clear_gpu_memory()

if __name__ == "__main__":
    main()
