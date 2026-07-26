# 🧠 Efficient CoT Distillation in Small Language Models (SLMs)

> **Paper Title:** Efficient Chain-of-Thought (CoT) Distillation in Small Language Models: A Parameter-Efficient Approach to Enhanced Reasoning
> **Target Framework:** QLoRA + Unsloth / HuggingFace TRL
> **Execution Platform:** Kaggle Free GPU Tier

---

## 📁 Repository Structure

```
SLMs/
├── masterplan.md                    # Project blueprint and step-by-step roadmap
├── configs/
│   └── qlora_config.yaml            # Hyperparameters and experiment configurations
├── src/
│   ├── data_utils.py                # Dataset download, formatting, and split helpers
│   ├── model_utils.py               # (Phase 2) QLoRA loading & memory profiling
│   ├── eval_utils.py                # (Phase 2) Benchmark evaluation metrics (GSM8K, ARC, EM)
│   └── prompts.py                   # (Phase 2) Standardized prompt templates
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Phase 1: Data acquisition and preparation
│   ├── 02_baseline_evaluation.ipynb # Phase 2: Base model zero-shot benchmarks
│   ├── 03_qlora_training.ipynb      # Phase 3: Fast QLoRA fine-tuning execution
│   └── 04_post_training_eval.ipynb  # Phase 4: Comparative evaluation & plotting
├── data/                            # Processed train/test jsonl files
├── results/                         # Evaluation outputs, metrics, and JSON traces
└── paper/                           # LaTeX paper sources and figures
```

---

## 🚀 Kaggle Execution Instructions

1. **Push to GitHub**: Commit and push this codebase to your repository.
2. **Open Kaggle**: Create a new Notebook on Kaggle and select **GPU T4 x2** or **P100**.
3. **Clone Repository**: In the first cell of your Kaggle notebook, clone your repo:
   ```bash
   !git clone https://github.com/YOUR_USERNAME/SLMs.git
   %cd SLMs
   ```
4. **Run Notebooks**: Open and execute `notebooks/01_data_preparation.ipynb` to verify Phase 1 dataset setup.
