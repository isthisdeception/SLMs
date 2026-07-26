# 🧠 MasterPlan: Efficient Chain-of-Thought Distillation in Small Language Models

> **A Parameter-Efficient Approach to Enhanced Reasoning**
>
> Architectural Blueprint & Step-by-Step Workflow Guide

---

## 📋 Project Overview

| Attribute | Detail |
|---|---|
| **Paper Title** | *Efficient Chain-of-Thought (CoT) Distillation in Small Language Models: A Parameter-Efficient Approach to Enhanced Reasoning* |
| **Target Length** | 7–8 pages (two-column, conference format) |
| **Candidate Models** | `Qwen/Qwen2.5-3B` or `meta-llama/Llama-3.2-3B` |
| **Dataset** | 5,000-sample subset of `HuggingFaceH4/Bespoke-Stratos-17k` (primary) or `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` (fallback) |
| **Compute** | Kaggle Free Tier — Dual NVIDIA T4 16 GB or single P100 16 GB |
| **Time Budget** | ≤ 7 hours wall-clock (training + evaluation) |
| **Optimization** | QLoRA (4-bit NF4) via **Unsloth** (primary) / HuggingFace TRL + BitsAndBytes (fallback) |
| **Paper Format** | ACL 2024 / IEEE Conference (to be confirmed by user) |

---

## 📁 Planned Repository Structure

```
SLMs/
├── masterplan.md                    # ← You are here
├── literature_review.md             # Phase 5 output
├── paper/
│   ├── main.tex                     # Conference paper (LaTeX)
│   ├── acl2024.sty                  # Style file (or IEEE equivalent)
│   ├── references.bib               # BibTeX references
│   └── figures/                     # All plots and diagrams
│       ├── training_loss.png
│       ├── reasoning_accuracy.png
│       ├── memory_profile.png
│       └── architecture_diagram.png
├── notebooks/
│   ├── 01_data_preparation.ipynb    # Phase 1
│   ├── 02_baseline_evaluation.ipynb # Phase 2
│   ├── 03_qlora_training.ipynb      # Phase 3
│   └── 04_post_training_eval.ipynb  # Phase 4
├── src/
│   ├── data_utils.py                # Dataset loading, filtering, formatting
│   ├── model_utils.py               # Model loading, QLoRA config
│   ├── eval_utils.py                # Evaluation metrics & benchmarks
│   ├── training_config.py           # Hyperparameters & SFT config
│   └── prompts.py                   # Prompt templates for CoT evaluation
├── configs/
│   └── qlora_config.yaml            # Centralized hyperparameter config
├── results/
│   ├── baseline_results.json
│   ├── sft_results.json
│   └── comparison_table.csv
└── README.md
```

---

## 🔬 Experimental Design Summary

### Independent Variables
- Fine-tuning method: **Baseline (zero-shot)** vs. **QLoRA-SFT on CoT data**
- LoRA rank: `r ∈ {8, 16, 32}` (ablation)
- Dataset size: `N ∈ {1000, 2500, 5000}` (scaling analysis)

### Dependent Variables (Metrics)
| Metric | What It Measures | Tool |
|---|---|---|
| **Exact Match (EM)** | Final-answer correctness | Custom string matching |
| **Reasoning Step Accuracy** | Quality of intermediate CoT steps | LLM-as-judge or heuristic scoring |
| **GSM8K Accuracy** | Math reasoning benchmark | 200-sample subset evaluation |
| **ARC-Challenge Accuracy** | Science reasoning benchmark | 200-sample subset evaluation |
| **ROUGE-L** | Surface overlap of generated CoT vs. reference | `rouge_score` library |
| **Perplexity** | Language modeling quality | HuggingFace `evaluate` |
| **Trainable Params (%)** | Parameter efficiency | Model inspection |
| **Peak VRAM (GB)** | Memory efficiency | `torch.cuda.max_memory_allocated()` |
| **Training Time (min)** | Computational efficiency | Wall-clock measurement |

### Controls
- Same prompt template for baseline and fine-tuned model
- Same generation hyperparameters (`temperature=0.0`, `max_new_tokens=512`)
- Same evaluation subset (held-out 500 samples from the 5,000)
- Fixed random seed (`seed=42`)

---

## ⏱️ Time Budget Allocation (7-Hour Window)

| Phase | Estimated Time | Cumulative |
|---|---|---|
| Environment setup + data download | 15 min | 0:15 |
| Baseline evaluation (zero-shot) | 45 min | 1:00 |
| QLoRA fine-tuning (3 epochs, 4,500 train samples) | 2.5–3 hrs | 4:00 |
| Post-training evaluation | 45 min | 4:45 |
| Ablation runs (LoRA rank sweep) | 1.5 hrs | 6:15 |
| Buffer / re-runs | 45 min | 7:00 |

---

---

# Phase 1: Environment Setup & Dataset Preparation

## 🎯 Goal
Set up the complete Kaggle-compatible environment, download and preprocess the CoT reasoning dataset, and prepare train/validation/test splits ready for fine-tuning.

## 📝 Technical Plan

### 1.1 — Environment & Dependencies
- Install core libraries: `unsloth`, `trl`, `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `rouge_score`, `evaluate`, `wandb` (optional logging).
- Pin specific versions for reproducibility.
- Verify GPU availability and VRAM with `torch.cuda.get_device_properties()`.
- Set all random seeds for reproducibility (`torch`, `numpy`, `random`, `transformers`).

### 1.2 — Dataset Selection & Download
- **Primary**: `HuggingFaceH4/Bespoke-Stratos-17k` — This dataset contains high-quality CoT reasoning traces distilled from frontier models. Each sample has a multi-turn conversation with explicit `<think>` reasoning blocks.
- **Fallback**: `Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B` — Larger dataset with diverse reasoning tasks.
- Download using `datasets.load_dataset()`.

### 1.3 — Data Preprocessing Pipeline
1. **Filter**: Remove samples with empty reasoning traces or malformed conversations.
2. **Subset**: Randomly sample 5,000 high-quality examples (stratified by reasoning category if metadata is available).
3. **Format Conversion**: Convert each sample into the chat-template format expected by the target model:
   ```
   <|im_start|>system
   You are a helpful assistant that thinks step-by-step before answering.<|im_end|>
   <|im_start|>user
   {question}<|im_end|>
   <|im_start|>assistant
   <think>{chain_of_thought}</think>
   {final_answer}<|im_end|>
   ```
4. **Split**: 4,500 train / 500 test (90/10).
5. **Tokenization Analysis**: Compute token length statistics (mean, median, max, P95) to inform `max_seq_length` setting.
6. **Save**: Export processed splits as `.jsonl` files and HuggingFace `Dataset` objects.

### 1.4 — Output Artifacts
- `data/train.jsonl` (4,500 samples)
- `data/test.jsonl` (500 samples)
- `data/stats.json` (token length statistics)
- Notebook: `notebooks/01_data_preparation.ipynb`
- Python module: `src/data_utils.py`

## 🤖 Antigravity Prompt (Phase 1)

```
Act as an Expert AI/NLP Engineer. We are executing Phase 1 of our masterplan (see masterplan.md in the project root).

TASK: Create the complete data preparation pipeline for our CoT distillation project.

Create the following files:

1. `src/data_utils.py` — A Python module containing:
   - `setup_environment()`: Function that prints GPU info, sets random seeds (42), and verifies CUDA availability.
   - `load_and_filter_dataset(dataset_name="HuggingFaceH4/Bespoke-Stratos-17k", n_samples=5000)`: Downloads the dataset, filters out malformed samples (empty reasoning, missing fields), and returns a clean subset of exactly 5,000 samples.
   - `format_for_chat_template(sample, model_type="qwen")`: Converts a single raw sample into the Qwen2.5 chat template format with <think> reasoning blocks. Support both "qwen" and "llama" model_type options.
   - `prepare_splits(dataset, train_ratio=0.9, seed=42)`: Splits into train/test and applies the chat template formatting.
   - `compute_token_stats(dataset, tokenizer)`: Returns a dict of token length statistics (mean, median, max, P95, P99) for the formatted text.
   - `save_splits(train_ds, test_ds, output_dir="data/")`: Saves as .jsonl files.

2. `notebooks/01_data_preparation.ipynb` — A Kaggle-ready Jupyter notebook that:
   - Installs all dependencies (unsloth, trl, transformers, datasets, peft, bitsandbytes, accelerate, rouge_score, evaluate) with pinned versions.
   - Imports and calls all functions from data_utils.py.
   - Displays sample examples from the processed dataset.
   - Shows token length distribution statistics.
   - Saves the processed train/test splits.
   - Has clear markdown headers and explanatory cells.

3. `configs/qlora_config.yaml` — A centralized config file with ALL hyperparameters we will use across the project:
   - model_name, dataset_name, n_samples, max_seq_length, lora_r, lora_alpha, lora_dropout, learning_rate, num_epochs, batch_size, gradient_accumulation_steps, warmup_ratio, weight_decay, seed, output_dir, etc.

IMPORTANT CONSTRAINTS:
- Everything must run on Kaggle free GPU (T4 16GB or P100).
- Use Unsloth's FastLanguageModel for model/tokenizer loading where applicable.
- max_seq_length should be set based on P95 token length (cap at 2048).
- All code must be production-quality with docstrings, type hints, and error handling.
- Use the Qwen2.5-3B model as the primary target (with Llama-3.2-3B as fallback).
```

---

---

# Phase 2: Model Setup, QLoRA Configuration & Baseline Evaluation

## 🎯 Goal
Load the base model with 4-bit quantization, configure QLoRA adapters, and run a comprehensive **zero-shot baseline evaluation** to establish the "before fine-tuning" performance numbers.

## 📝 Technical Plan

### 2.1 — Model Loading with Unsloth
- Use `FastLanguageModel.from_pretrained()` with:
  - `model_name = "unsloth/Qwen2.5-3B"` (or `"unsloth/Llama-3.2-3B"`)
  - `max_seq_length = 2048`
  - `load_in_4bit = True` (NF4 quantization)
  - `dtype = None` (auto-detect: float16 on T4, bfloat16 on A100+)
- Log model footprint: total params, trainable params, VRAM usage.

### 2.2 — QLoRA Adapter Configuration
- Apply LoRA adapters via `FastLanguageModel.get_peft_model()`:
  ```python
  model = FastLanguageModel.get_peft_model(
      model,
      r=16,                          # LoRA rank
      lora_alpha=32,                 # Alpha = 2*r (standard practice)
      lora_dropout=0.05,
      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
      bias="none",
      use_gradient_checkpointing="unsloth",  # 60% less VRAM
      random_state=42,
  )
  ```
- Report: trainable parameters count, percentage of total, estimated VRAM.

### 2.3 — Baseline Evaluation (Zero-Shot)
- Evaluate the **unmodified base model** (no LoRA adapters active) on:
  1. **Held-out test set** (500 samples from our CoT dataset):
     - Generate CoT responses with `temperature=0.0, max_new_tokens=512`.
     - Measure: Exact Match on final answer, ROUGE-L on reasoning trace.
  2. **GSM8K subset** (200 random problems):
     - Standard few-shot (0-shot and 8-shot) math reasoning evaluation.
     - Extract final numerical answer, compute accuracy.
  3. **ARC-Challenge subset** (200 random questions):
     - Multiple-choice science reasoning.
     - Compute accuracy.
- Save all results to `results/baseline_results.json`.
- Generate qualitative examples: 5 sample outputs showing the model's reasoning attempts.

### 2.4 — Output Artifacts
- `src/model_utils.py` (model loading, QLoRA setup)
- `src/eval_utils.py` (evaluation pipelines)
- `src/prompts.py` (prompt templates)
- `notebooks/02_baseline_evaluation.ipynb`
- `results/baseline_results.json`

## 🤖 Antigravity Prompt (Phase 2)

```
Act as an Expert AI/NLP Engineer. We are executing Phase 2 of our masterplan (see masterplan.md).

TASK: Create the model loading infrastructure, QLoRA configuration, and baseline evaluation pipeline.

Create the following files:

1. `src/model_utils.py` — A Python module containing:
   - `load_base_model(model_name="unsloth/Qwen2.5-3B", max_seq_length=2048, load_in_4bit=True)`: Loads the model and tokenizer using Unsloth's FastLanguageModel. Returns (model, tokenizer). Logs total params, VRAM usage.
   - `apply_qlora(model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None)`: Applies QLoRA adapters using Unsloth. Returns the PEFT model. Logs trainable params count and percentage.
   - `get_memory_stats()`: Returns a dict with current/peak VRAM usage in GB.
   - `save_model(model, tokenizer, output_dir)`: Saves the fine-tuned LoRA adapters and tokenizer.

2. `src/eval_utils.py` — A Python module containing:
   - `generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.0)`: Generates a single response using greedy decoding.
   - `batch_generate(model, tokenizer, prompts, batch_size=4, max_new_tokens=512)`: Batch generation for efficiency.
   - `extract_final_answer(response_text)`: Extracts the final answer from a CoT response (handles both boxed answers and plain text).
   - `extract_thinking(response_text)`: Extracts the <think>...</think> reasoning block.
   - `compute_exact_match(predictions, references)`: Computes EM accuracy.
   - `compute_rouge_l(predictions, references)`: Computes ROUGE-L F1 scores.
   - `evaluate_gsm8k(model, tokenizer, n_samples=200, n_shot=0)`: Evaluates on GSM8K subset. Returns accuracy and per-sample results.
   - `evaluate_arc_challenge(model, tokenizer, n_samples=200)`: Evaluates on ARC-Challenge subset. Returns accuracy.
   - `evaluate_cot_dataset(model, tokenizer, test_dataset, n_samples=None)`: Full evaluation on our held-out CoT test set. Returns dict with EM, ROUGE-L, and qualitative examples.
   - `run_full_baseline(model, tokenizer, test_dataset)`: Orchestrates all evaluations and returns a comprehensive results dict.
   - `save_results(results, filepath)`: Saves results as formatted JSON.

3. `src/prompts.py` — Prompt templates:
   - `COT_SYSTEM_PROMPT`: System prompt for CoT reasoning.
   - `format_cot_prompt(question, model_type="qwen")`: Formats a question with the appropriate chat template.
   - `format_gsm8k_prompt(question, n_shot=0)`: Formats GSM8K with optional few-shot examples.
   - `format_arc_prompt(question, choices)`: Formats ARC multiple-choice questions.

4. `notebooks/02_baseline_evaluation.ipynb` — A Kaggle-ready notebook that:
   - Loads the base model with 4-bit quantization using Unsloth.
   - Reports model size, trainable params, VRAM usage.
   - Configures QLoRA adapters and reports the parameter efficiency.
   - Runs the FULL baseline evaluation (CoT test set, GSM8K-200, ARC-200).
   - Displays 5 qualitative examples of model outputs.
   - Saves all results to results/baseline_results.json.
   - Has clear markdown documentation throughout.

IMPORTANT CONSTRAINTS:
- All evaluation must use greedy decoding (temperature=0.0) for reproducibility.
- GSM8K and ARC evaluation should load subsets to stay within time budget (~45 min total for baseline).
- Handle OOM gracefully — if a sample is too long, skip it and log a warning.
- Use torch.cuda.empty_cache() between evaluation stages.
- The baseline model should be evaluated WITHOUT any LoRA adapters applied.
```

---

---

# Phase 3: QLoRA Fine-Tuning for CoT Reasoning

## 🎯 Goal
Fine-tune the base model on 4,500 CoT reasoning samples using QLoRA + Unsloth, producing a model that generates structured chain-of-thought reasoning. Must complete within ~3 hours on a T4/P100.

## 📝 Technical Plan

### 3.1 — Training Configuration
| Hyperparameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 2e-4 | Standard for QLoRA (Dettmers et al., 2023) |
| `lr_scheduler_type` | cosine | Smooth decay, better convergence |
| `num_train_epochs` | 3 | Balance quality vs. time budget |
| `per_device_train_batch_size` | 2 | Fits in T4 16GB with 4-bit |
| `gradient_accumulation_steps` | 4 | Effective batch size = 8 |
| `warmup_ratio` | 0.05 | ~67 warmup steps on 4,500 samples |
| `weight_decay` | 0.01 | Standard regularization |
| `max_seq_length` | 2048 | Based on P95 token length from Phase 1 |
| `fp16` | True (T4) / `bf16` True (A100) | Mixed precision training |
| `gradient_checkpointing` | True (Unsloth mode) | 60% VRAM reduction |
| `packing` | True | Pack short sequences for efficiency |
| `logging_steps` | 10 | Frequent loss logging |
| `save_strategy` | "steps" | Save checkpoints at 500 steps |
| `eval_strategy` | "steps" | Evaluate every 250 steps |

### 3.2 — Training Pipeline (Unsloth + TRL SFTTrainer)
1. Load preprocessed data from Phase 1.
2. Load model + QLoRA adapters from Phase 2 config.
3. Initialize `SFTTrainer` with the above hyperparameters.
4. Set up callbacks: `EarlyStoppingCallback(patience=3)`, custom VRAM logging callback.
5. Train. Monitor loss curve, VRAM, and throughput (tokens/sec).
6. Save the best checkpoint (lowest eval loss).

### 3.3 — Training Monitoring
- Log every 10 steps: `train_loss`, `learning_rate`, `gpu_memory_allocated`.
- Log every 250 steps: `eval_loss`, sample generation from 3 fixed prompts.
- Estimated throughput: ~1,500–2,500 tokens/sec with Unsloth on T4.
- Estimated total training time: 2–3 hours for 3 epochs on 4,500 samples.

### 3.4 — Ablation Runs (Time Permitting)
If time remains after the primary training run:
- **LoRA Rank Ablation**: Train with `r=8` and `r=32` (in addition to the primary `r=16`), 1 epoch each.
- **Dataset Size Ablation**: Train with `N=1000` and `N=2500`, 1 epoch each.
- These are fast runs (~20–40 min each) and provide valuable data for the paper.

### 3.5 — Output Artifacts
- `src/training_config.py` (training config and SFTTrainer setup)
- `notebooks/03_qlora_training.ipynb`
- `models/qwen2.5-3b-cot-qlora/` (saved LoRA adapters)
- `results/training_logs.json` (loss curves, VRAM traces)

## 🤖 Antigravity Prompt (Phase 3)

```
Act as an Expert AI/NLP Engineer. We are executing Phase 3 of our masterplan (see masterplan.md).

TASK: Create the complete QLoRA fine-tuning pipeline for CoT reasoning distillation.

Create the following files:

1. `src/training_config.py` — A Python module containing:
   - `get_training_args(output_dir="models/qwen2.5-3b-cot-qlora", num_epochs=3, batch_size=2, grad_accum=4, lr=2e-4, max_seq_length=2048)`: Returns a fully configured `TrainingArguments` (or `SFTConfig`) object optimized for T4 16GB. Include cosine LR schedule, warmup_ratio=0.05, weight_decay=0.01, fp16=True, gradient_checkpointing, logging every 10 steps, eval every 250 steps, save every 500 steps.
   - `get_sft_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, max_seq_length=2048, packing=True)`: Returns a configured SFTTrainer from TRL with the data collator, formatting function, and callbacks (EarlyStopping with patience=3, custom VRAMLoggingCallback).
   - `VRAMLoggingCallback(TrainerCallback)`: Custom callback that logs GPU memory usage at each logging step.
   - `run_training(model, tokenizer, train_dataset, eval_dataset, **kwargs)`: End-to-end training orchestration function. Returns the trainer and training results.

2. `notebooks/03_qlora_training.ipynb` — A Kaggle-ready notebook that:
   - Cell 1: Install dependencies (same as Phase 1).
   - Cell 2: Load processed data from Phase 1 (or re-run data prep if files not found).
   - Cell 3: Load model and apply QLoRA adapters (from Phase 2 utilities).
   - Cell 4: Configure training arguments. Print the full config for reproducibility.
   - Cell 5: Initialize SFTTrainer with all callbacks.
   - Cell 6: RUN TRAINING. Display progress with loss logging.
   - Cell 7: Plot training loss curve and learning rate schedule.
   - Cell 8: Plot VRAM usage over training.
   - Cell 9: Save the best model checkpoint (LoRA adapters only).
   - Cell 10: Generate 5 sample outputs from the fine-tuned model to verify quality.
   - Cell 11: (Optional) Run LoRA rank ablation (r=8, r=32) — 1 epoch each, if time permits.
   - Cell 12: Save all training logs and plots.
   - Full markdown documentation in every cell.

IMPORTANT CONSTRAINTS:
- Training MUST complete within 3 hours on a T4 16GB GPU.
- Use Unsloth's FastLanguageModel and optimized kernels for maximum throughput.
- Enable gradient checkpointing in "unsloth" mode to minimize VRAM.
- Use packing=True for sequence packing efficiency.
- Save ONLY the LoRA adapter weights (not the full model) to minimize storage.
- Implement graceful OOM handling — if training crashes, catch the error, reduce batch size, and retry.
- All plots should be publication-quality (matplotlib with clean styling, proper labels, saved as high-res PNGs).
- Print estimated remaining time at each logging step.
```

---

---

# Phase 4: Post-Training Evaluation

## 🎯 Goal
Comprehensively evaluate the fine-tuned model against the baseline, generating all quantitative results, qualitative examples, and publication-ready figures for the paper.

## 📝 Technical Plan

### 4.1 — Quantitative Evaluation (Same Pipeline as Baseline)
Run the exact same evaluation suite from Phase 2 on the **fine-tuned model**:
1. **CoT Test Set** (500 samples): Exact Match, ROUGE-L, reasoning quality.
2. **GSM8K** (200 samples): 0-shot math reasoning accuracy.
3. **ARC-Challenge** (200 samples): Science reasoning accuracy.

### 4.2 — Comparative Analysis
- Compute deltas: `Δ = SFT_metric - Baseline_metric` for every metric.
- Statistical significance: Bootstrap confidence intervals (95% CI) on accuracy metrics.
- Effect size: Cohen's d for key metrics.

### 4.3 — Ablation Results
- Compile results from LoRA rank sweep (`r ∈ {8, 16, 32}`).
- Compile results from dataset size sweep (`N ∈ {1000, 2500, 5000}`).
- Create a "parameter efficiency vs. performance" analysis.

### 4.4 — Qualitative Analysis
- **Side-by-side comparison**: 5 examples showing baseline vs. fine-tuned responses.
- **Error analysis**: Categorize failure modes (arithmetic errors, logic errors, hallucinated steps, premature termination).
- **Reasoning depth**: Measure average number of reasoning steps before vs. after fine-tuning.

### 4.5 — Publication-Ready Figures & Tables
Generate the following (saved as high-res PNGs and LaTeX tables):
1. **Table 1**: Main results comparison (Baseline vs. SFT across all metrics).
2. **Table 2**: Ablation results (LoRA rank, dataset size).
3. **Table 3**: Efficiency metrics (trainable params, VRAM, training time).
4. **Figure 1**: Training loss curve (with eval loss overlay).
5. **Figure 2**: Bar chart comparing accuracy across benchmarks.
6. **Figure 3**: LoRA rank vs. accuracy trade-off plot.
7. **Figure 4**: Dataset size scaling curve.
8. **Figure 5**: VRAM usage profile during training.

### 4.6 — Output Artifacts
- `notebooks/04_post_training_eval.ipynb`
- `results/sft_results.json`
- `results/comparison_table.csv`
- `results/ablation_results.json`
- `paper/figures/` (all publication figures)

## 🤖 Antigravity Prompt (Phase 4)

```
Act as an Expert AI/NLP Engineer. We are executing Phase 4 of our masterplan (see masterplan.md).

TASK: Run comprehensive post-training evaluation and generate all publication-ready results.

Create the following files:

1. `notebooks/04_post_training_eval.ipynb` — A Kaggle-ready notebook that:
   - Loads the fine-tuned model (base model + saved LoRA adapters).
   - Runs the EXACT same evaluation suite as the baseline (Phase 2):
     a. CoT Test Set (500 samples): EM accuracy, ROUGE-L on reasoning traces.
     b. GSM8K (200 samples): 0-shot accuracy with answer extraction.
     c. ARC-Challenge (200 samples): Multiple-choice accuracy.
   - Loads baseline results from results/baseline_results.json.
   - Computes all deltas (SFT - Baseline) and percentage improvements.
   - Computes bootstrap 95% confidence intervals for accuracy metrics (1000 resamples).
   - Generates side-by-side qualitative comparison (5 examples: baseline vs. SFT output).
   - Performs error analysis: categorize failures into types (arithmetic, logic, hallucination, incomplete reasoning).
   - Measures reasoning depth: average number of reasoning steps (sentences in <think> block).
   - If ablation results exist, compiles LoRA rank and dataset size comparisons.
   - Generates ALL publication-ready figures:
     * Figure 1: Training loss + eval loss curves (dual y-axis if needed).
     * Figure 2: Grouped bar chart — Baseline vs. SFT accuracy across benchmarks (GSM8K, ARC, CoT-EM).
     * Figure 3: LoRA rank vs. accuracy (line plot with error bars).
     * Figure 4: Dataset size scaling curve (log-x axis).
     * Figure 5: VRAM usage profile.
   - Generates LaTeX-formatted tables:
     * Table 1: Main results (copy-pasteable into LaTeX).
     * Table 2: Ablation results.
     * Table 3: Efficiency comparison.
   - Saves everything to results/ and paper/figures/.

2. `results/comparison_table.csv` — Machine-readable results for all metrics.

IMPORTANT CONSTRAINTS:
- All figures must be publication-quality: use matplotlib with a clean academic style (seaborn "paper" context, appropriate font sizes for two-column papers, high DPI=300).
- Use consistent color palette across all figures (e.g., baseline=gray, SFT=blue).
- LaTeX table output must be directly copy-pasteable into an ACL/IEEE paper.
- Include p-values or confidence intervals for key claims.
- Qualitative examples should be formatted as clean side-by-side comparisons.
- Save all figures as both PNG (300 DPI) and PDF (vector) for the paper.
```

---

---

# Phase 5: Literature Review & Related Work

## 🎯 Goal
Produce a comprehensive, well-cited literature review covering chain-of-thought reasoning, knowledge distillation, parameter-efficient fine-tuning, and small language models. This will form Section 2 of the paper and also serve as a standalone deliverable.

## 📝 Technical Plan

### 5.1 — Topic Areas to Cover

#### A. Chain-of-Thought Reasoning (2022–2026)
- **Foundational**: Wei et al. (2022) — CoT prompting; Kojima et al. (2022) — Zero-shot CoT ("Let's think step by step").
- **Advanced CoT**: Wang et al. (2023) — Self-consistency; Yao et al. (2023) — Tree of Thoughts; Besta et al. (2024) — Graph of Thoughts.
- **CoT in Small Models**: Magister et al. (2023) — Teaching small LMs to reason; Fu et al. (2023) — Specializing smaller models; Shridhar et al. (2023) — Distilling reasoning capabilities.
- **2024–2026 Advances**: DeepSeek-R1 distillation (2025); Qwen2.5 technical report; reasoning traces as training data.

#### B. Knowledge Distillation for LLMs
- Hinton et al. (2015) — Original KD framework.
- Hsieh et al. (2023) — Distilling step-by-step.
- Mukherjee et al. (2023) — Orca: Learning from complex explanation traces.
- West et al. (2023) — Symbolic knowledge distillation.
- 2024–2025: Dataset distillation approaches (Bespoke-Stratos, Magpie-Align).

#### C. Parameter-Efficient Fine-Tuning (PEFT)
- Hu et al. (2022) — LoRA.
- Dettmers et al. (2023) — QLoRA.
- Liu et al. (2024) — DoRA: Weight-decomposed low-rank adaptation.
- Renduchintala et al. (2024) — Tied-LoRA.
- Unsloth (2024) — Optimized training kernels.

#### D. Small Language Models (SLMs)
- Phi-1/2/3/4 (Microsoft) — Textbook-quality data for small models.
- Qwen2/2.5 series (Alibaba) — Architecture and capabilities.
- Llama 3/3.1/3.2 (Meta) — Scaling down from 405B to 1B.
- Gemma 2/3 (Google) — Efficient small model design.
- SmolLM, TinyLlama, etc.

### 5.2 — Literature Review Structure (for `literature_review.md`)
1. Introduction to CoT reasoning and its importance
2. Evolution of chain-of-thought techniques
3. Knowledge distillation: From classic to LLM-era
4. Parameter-efficient fine-tuning methods
5. Small language models: Capabilities and limitations
6. Research gap and our contribution
7. Complete reference list (BibTeX-ready)

### 5.3 — Methodology
- Search arXiv, Semantic Scholar, ACL Anthology for relevant papers.
- Focus on 2023–2026 publications for recency.
- Include seminal older works (2015–2022) for foundations.
- Target: 40–60 references total.

### 5.4 — Output Artifacts
- `literature_review.md` (comprehensive standalone document)
- `paper/references.bib` (all BibTeX entries)

## 🤖 Antigravity Prompt (Phase 5)

```
Act as an Expert AI Researcher specializing in NLP and Language Model Reasoning. We are executing Phase 5 of our masterplan (see masterplan.md).

TASK: Generate a comprehensive literature review and complete BibTeX reference file for our paper on "Efficient Chain-of-Thought Distillation in Small Language Models."

Create the following files:

1. `literature_review.md` — A comprehensive, well-structured literature review (3,000–4,000 words) covering:

   Section 1: Chain-of-Thought Reasoning
   - Foundational CoT work: Wei et al. (2022), Kojima et al. (2022) zero-shot CoT.
   - Advanced CoT: Self-consistency (Wang et al., 2023), Tree of Thoughts (Yao et al., 2023), Graph of Thoughts.
   - CoT in small models: Magister et al. (2023), Fu et al. (2023), Shridhar et al. (2023).
   - Latest 2024–2026: DeepSeek-R1 reasoning distillation, Qwen2.5 reasoning capabilities, o1/o3-style reasoning.

   Section 2: Knowledge Distillation for LLMs
   - Classic KD: Hinton et al. (2015).
   - LLM-era KD: Distilling step-by-step (Hsieh et al., 2023), Orca (Mukherjee et al., 2023).
   - Dataset distillation: Bespoke-Stratos, Magpie-Align, OpenHermes, UltraChat.
   - Reasoning trace distillation from frontier models (GPT-4, DeepSeek-R1, Claude).

   Section 3: Parameter-Efficient Fine-Tuning
   - LoRA (Hu et al., 2022), QLoRA (Dettmers et al., 2023).
   - Recent advances: DoRA, Tied-LoRA, LoRA+ (2024).
   - Unsloth and efficient training frameworks.
   - Comparative analysis of PEFT methods.

   Section 4: Small Language Models
   - Phi series (Microsoft), Qwen2.5 series (Alibaba), Llama 3.2 (Meta), Gemma (Google).
   - Capabilities vs. limitations of 1B–3B parameter models.
   - Reasoning capabilities of SLMs vs. large models.

   Section 5: Research Gap & Our Contribution
   - Identify the specific gap: lack of systematic study on CoT distillation into SLMs using parameter-efficient methods.
   - Position our contribution: combining QLoRA + CoT distillation + systematic evaluation on 3B models.

   For each paper cited, include: authors, year, key finding, and relevance to our work.
   Write in formal academic prose suitable for a top-tier NLP conference.

2. `paper/references.bib` — Complete BibTeX file with ALL referenced papers (target: 40-60 entries). Each entry must have:
   - Correct BibTeX key format (AuthorYear, e.g., wei2022chain).
   - Complete fields: author, title, booktitle/journal, year, pages (if available), url/doi.
   - Accurate publication venues.

IMPORTANT CONSTRAINTS:
- Focus on papers from 2022–2026, with seminal older works included.
- Ensure factual accuracy — do NOT hallucinate paper titles, authors, or findings. If uncertain about a specific detail, note it with [VERIFY].
- Use formal academic writing style (third person, passive voice where appropriate).
- The literature review should clearly build a narrative leading to our research contribution.
- BibTeX entries must be syntactically valid and compile without errors.
- Cross-reference the literature review sections with the BibTeX keys used.
```

---

---

# Phase 6: Drafting the Conference Paper

## 🎯 Goal
Write a complete 7–8 page conference paper in LaTeX, incorporating all experimental results from Phases 2–4 and the literature review from Phase 5.

## 📝 Technical Plan

### 6.1 — Paper Structure (ACL/IEEE Format)

| Section | Pages | Content |
|---|---|---|
| **Title + Authors + Abstract** | 0.5 | 150–250 word abstract summarizing contribution |
| **1. Introduction** | 1.0 | Motivation, problem statement, contributions (3 bullet points), paper outline |
| **2. Related Work** | 1.0–1.5 | Condensed from literature_review.md, organized by theme |
| **3. Methodology** | 1.5–2.0 | Model selection, QLoRA setup, dataset, training procedure |
| **4. Experimental Setup** | 0.5 | Benchmarks, metrics, hardware, baselines |
| **5. Results & Analysis** | 1.5–2.0 | Main results table, ablations, qualitative analysis, error analysis |
| **6. Discussion** | 0.5 | Implications, limitations, comparison with concurrent work |
| **7. Conclusion** | 0.3 | Summary, future work |
| **References** | ~1.0 | 30–50 references |

### 6.2 — Key Claims & Contributions
1. We demonstrate that QLoRA fine-tuning on distilled CoT data significantly improves reasoning capabilities of 3B-parameter models.
2. We provide a systematic analysis of parameter-efficient CoT distillation, including LoRA rank and dataset size ablations.
3. We show that this approach is practically feasible on consumer-grade hardware (single T4 GPU, <3 hours training).

### 6.3 — Writing Guidelines
- **Abstract**: Problem → Approach → Key Results → Significance.
- **Introduction**: Hook → Problem → Gap → Our Approach → Contributions → Outline.
- **Related Work**: Group by theme, not chronologically. End each paragraph with a transition to our work.
- **Methodology**: Be precise enough for reproducibility. Include all hyperparameters.
- **Results**: Lead with the main finding, then support with details. Every table/figure must be referenced in text.
- **Discussion**: Honest about limitations. Compare fairly with concurrent work.

### 6.4 — Output Artifacts
- `paper/main.tex`
- `paper/acl2024.sty` (or IEEE template)
- Updated `paper/references.bib`
- All figures already in `paper/figures/`

## 🤖 Antigravity Prompt (Phase 6)

```
Act as an Expert AI Researcher and Academic Writer. We are executing Phase 6 of our masterplan (see masterplan.md).

TASK: Draft the complete conference paper in LaTeX format.

You have access to:
- Experimental results in results/baseline_results.json, results/sft_results.json, results/comparison_table.csv, and results/ablation_results.json.
- Literature review in literature_review.md.
- All figures in paper/figures/.
- BibTeX references in paper/references.bib.

Create the following files:

1. `paper/main.tex` — A complete, compilable LaTeX paper using [ACL 2024 / IEEE conference — to be confirmed] format:

   STRUCTURE:
   - Title: "Efficient Chain-of-Thought Distillation in Small Language Models: A Parameter-Efficient Approach to Enhanced Reasoning"
   - Abstract (150–250 words): Problem → Approach → Key Results (with specific numbers) → Significance.
   - Section 1 — Introduction (1 page):
     * Hook: The power of CoT reasoning and its limitation to large models.
     * Problem: Small models struggle with multi-step reasoning.
     * Gap: Limited systematic study of CoT distillation into SLMs using PEFT.
     * Our approach: QLoRA + distilled CoT data on Qwen2.5-3B.
     * Contributions: 3 numbered bullet points.
     * Paper outline.
   - Section 2 — Related Work (1–1.5 pages):
     * 2.1 Chain-of-Thought Reasoning
     * 2.2 Knowledge Distillation for Language Models
     * 2.3 Parameter-Efficient Fine-Tuning
     * 2.4 Small Language Models
     * Condense from literature_review.md. Each subsection: 2–3 paragraphs.
   - Section 3 — Methodology (1.5–2 pages):
     * 3.1 Model Architecture (Qwen2.5-3B details).
     * 3.2 QLoRA Configuration (quantization, LoRA params, target modules).
     * 3.3 Dataset (source, size, preprocessing, format with example).
     * 3.4 Training Procedure (optimizer, LR schedule, all hyperparameters as a table).
   - Section 4 — Experimental Setup (0.5 pages):
     * Benchmarks (CoT test set, GSM8K, ARC-Challenge).
     * Metrics (EM, ROUGE-L, accuracy).
     * Hardware and compute budget.
     * Baselines.
   - Section 5 — Results & Analysis (1.5–2 pages):
     * 5.1 Main Results (Table 1 + discussion).
     * 5.2 Ablation Studies (Table 2 + Figures 3, 4).
     * 5.3 Efficiency Analysis (Table 3 + Figure 5).
     * 5.4 Qualitative Analysis (Figure/table with side-by-side examples).
     * 5.5 Error Analysis (categorized failure modes).
   - Section 6 — Discussion (0.5 pages):
     * Key findings and implications.
     * Limitations (dataset size, single model family, benchmark coverage).
     * Comparison with concurrent work.
   - Section 7 — Conclusion & Future Work (0.3 pages).
   - References.

   USE PLACEHOLDER VALUES like [XX.X] where actual experimental numbers are needed — we will fill these in after running experiments. But structure ALL tables and figures as if they will contain real data.

2. `paper/acl2024.sty` — The official ACL 2024 style file (or provide instructions to download it). If using IEEE, use the appropriate IEEEtran class instead.

IMPORTANT CONSTRAINTS:
- Paper MUST fit within 7–8 pages (excluding references).
- Every figure and table must be referenced in the text with \ref{}.
- Use \citep{} and \citet{} properly for citations.
- All hyperparameters must appear in a table for reproducibility.
- Writing must be formal, precise, and suitable for a top-tier venue (ACL, EMNLP, or IEEE equivalent).
- Include appropriate \usepackage commands for all needed packages.
- The paper must compile with pdflatex without errors (given the style file).
- Use booktabs for tables (\toprule, \midrule, \bottomrule).
```

---

---

# Phase 7: Final Review & Formatting

## 🎯 Goal
Polish the paper to publication quality, ensure it meets page limits, fix any formatting issues, and prepare the final submission package.

## 📝 Technical Plan

### 7.1 — Content Review Checklist
- [ ] Abstract accurately reflects final results (update placeholder numbers).
- [ ] All experimental numbers in text match tables/figures.
- [ ] Every claim is supported by evidence (table, figure, or citation).
- [ ] Contributions in introduction match what was actually achieved.
- [ ] Related work is fair and comprehensive.
- [ ] Methodology is detailed enough for reproducibility.
- [ ] Limitations are honestly discussed.
- [ ] Future work suggestions are concrete and feasible.

### 7.2 — Formatting Review Checklist
- [ ] Paper fits within 7–8 pages (main content, excluding references).
- [ ] All figures are high resolution (300+ DPI) and readable in print.
- [ ] All tables use `booktabs` formatting.
- [ ] Font sizes in figures match body text.
- [ ] No orphan/widow lines.
- [ ] All citations are complete and correctly formatted.
- [ ] No compilation warnings or errors.
- [ ] Supplementary materials are referenced if included.

### 7.3 — Writing Quality Review
- [ ] No grammatical errors or typos.
- [ ] Consistent terminology throughout (e.g., "fine-tuning" vs "finetuning").
- [ ] Active voice where possible.
- [ ] Each paragraph has a clear topic sentence.
- [ ] Smooth transitions between sections.
- [ ] No unsupported superlatives ("significantly" requires statistical significance).

### 7.4 — Final Deliverables Package
```
submission/
├── main.pdf                    # Compiled paper
├── main.tex                    # LaTeX source
├── references.bib              # Bibliography
├── acl2024.sty                 # Style file
├── figures/                    # All figure files
├── supplementary/
│   ├── appendix.tex            # Additional experiments, full prompt templates
│   ├── hyperparameters.tex     # Complete hyperparameter table
│   └── code_availability.md    # Link to code repository
└── README.md                   # Submission instructions
```

## 🤖 Antigravity Prompt (Phase 7)

```
Act as an Expert Academic Editor and LaTeX Specialist. We are executing Phase 7 (final phase) of our masterplan (see masterplan.md).

TASK: Perform a comprehensive final review and polish of our conference paper.

Do the following:

1. CONTENT REVIEW:
   - Read through paper/main.tex completely.
   - Verify that all placeholder values [XX.X] have been replaced with actual experimental numbers from results/.
   - Check that every claim in the paper is supported by a table, figure, or citation.
   - Verify that the abstract accurately summarizes the final results.
   - Ensure the contributions listed in the introduction match the actual results.
   - Check that related work citations match entries in references.bib.

2. FORMATTING REVIEW:
   - Verify the paper fits within the 7–8 page limit (main content only).
   - If over the limit: suggest specific cuts (least important content to remove).
   - If under the limit: suggest specific additions (more analysis, examples, or discussion).
   - Check all figures are properly referenced with \ref{} and have descriptive captions.
   - Verify all tables use booktabs formatting.
   - Check for orphan/widow lines and fix with manual page breaks if needed.

3. WRITING QUALITY:
   - Fix any grammatical errors, typos, or awkward phrasing.
   - Ensure consistent terminology throughout.
   - Verify all mathematical notation is consistent.
   - Check that every acronym is defined on first use.

4. BIBLIOGRAPHY:
   - Verify all \cite{} commands resolve to entries in references.bib.
   - Check for missing or incomplete BibTeX entries.
   - Ensure citation style is consistent (author-year or numbered, per conference style).

5. FINAL OUTPUT:
   - Update paper/main.tex with all fixes and polish.
   - Create a `paper/submission_checklist.md` with a completed checklist of all items verified.
   - Create `paper/supplementary/appendix.tex` with:
     * Full prompt templates used.
     * Complete hyperparameter table.
     * Additional qualitative examples (5–10 more).
     * Per-category breakdown of results.
   - Create a README.md in the project root summarizing the entire project, how to reproduce results, and the paper abstract.

IMPORTANT CONSTRAINTS:
- Do NOT change the core experimental claims or results — only polish the presentation.
- Maintain the current section structure unless a change is clearly needed for flow.
- All LaTeX changes must compile without errors.
- The final paper should read as a polished, camera-ready submission to a top NLP venue.
```

---

---

# 📊 Success Criteria

| Criterion | Target |
|---|---|
| Training completes on T4/P100 | ✅ Within 3 hours |
| Total compute time (all phases) | ✅ Within 7 hours |
| Peak VRAM usage | ✅ < 15 GB |
| CoT reasoning improvement (EM) | ✅ ≥ 10% absolute improvement over baseline |
| GSM8K accuracy improvement | ✅ ≥ 5% absolute improvement |
| Trainable parameters | ✅ < 2% of total model parameters |
| Paper length | ✅ 7–8 pages (excluding references) |
| References | ✅ 30–50 cited works |
| All code runs on Kaggle | ✅ No errors, no OOM |

---

# 🚨 Risk Mitigation

| Risk | Mitigation |
|---|---|
| OOM on T4 | Reduce `max_seq_length` to 1024, reduce batch size to 1, use gradient checkpointing |
| Training too slow | Reduce epochs to 2, reduce dataset to 3,000 samples, skip ablations |
| Poor baseline performance | Switch from Qwen2.5-3B to Llama-3.2-3B (or vice versa) |
| Dataset quality issues | Switch to fallback dataset (Magpie-Align), add quality filtering |
| Unsloth incompatibility | Fall back to HuggingFace TRL + BitsAndBytes (slightly slower but compatible) |
| Marginal improvements | Expand analysis to include perplexity improvements, reasoning depth, qualitative gains |
| Paper too long/short | Adjust discussion depth, add/remove ablations, expand/trim related work |

---

# 📅 Execution Timeline

```
Day 1: Phase 1 + Phase 2 (Setup + Baseline)          → ~2 hours
Day 1: Phase 3 (Training)                              → ~3 hours (can run while writing)
Day 1: Phase 4 (Evaluation)                             → ~1.5 hours
Day 2: Phase 5 (Literature Review)                      → ~2 hours
Day 2: Phase 6 (Paper Draft)                            → ~3 hours
Day 3: Phase 7 (Final Review)                           → ~2 hours
                                                        ──────────
Total:                                                  ~13.5 hours of work
```

> **Note**: Phases 5 and 6 (writing) can be done in parallel with Phases 3 and 4 (compute) since they are independent tasks.

---

## ❓ Questions Before We Begin

Before we start executing Phase 1, please clarify:

1. **Target Conference**: Do you have a specific target venue (ACL, EMNLP, NAACL, IEEE, AAAI, or other)? This determines the LaTeX template and formatting guidelines.
2. **University Guidelines**: Do you have any specific university formatting requirements, page limits, or submission rules that override conference defaults?
3. **Author Information**: How should the author block be formatted? (Names, affiliations, emails)
4. **Model Preference**: Do you have a preference between **Qwen2.5-3B** and **Llama-3.2-3B**, or should we default to Qwen2.5-3B?
5. **Kaggle Account**: Is your Kaggle account already set up with GPU access enabled?

---

*This masterplan was generated as the architectural blueprint for the CoT Distillation research project. Execute each phase sequentially by pasting the corresponding Antigravity Prompt.*
