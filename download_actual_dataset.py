"""
Downloads and saves the EXACT 5,000-sample dataset (train.jsonl and test.jsonl)
used in this research directly to your local computer.
"""

import os
import sys
import yaml

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_utils import (
    load_and_filter_dataset,
    prepare_splits,
    save_splits
)

def main():
    print("==========================================================")
    print("Downloading the ACTUAL 5,000-sample dataset used in research...")
    print("==========================================================")
    
    # Load config settings
    config_path = os.path.join(PROJECT_ROOT, "configs", "qlora_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Download & Filter 5,000 reasoning samples from HuggingFaceH4/Bespoke-Stratos-17k
    raw_ds = load_and_filter_dataset(
        dataset_name=config['data']['dataset_name'],
        fallback_dataset_name=config['data']['fallback_dataset_name'],
        n_samples=config['data']['n_samples'],
        seed=config['project']['seed']
    )

    # 2. Convert to ChatML template (<think> blocks) and create 4,500 train / 500 test splits
    train_ds, test_ds = prepare_splits(
        dataset=raw_ds,
        train_ratio=config['data']['train_ratio'],
        seed=config['project']['seed'],
        model_type=config['model']['model_type']
    )

    # 3. Save locally into data/train.jsonl and data/test.jsonl
    data_dir = os.path.join(PROJECT_ROOT, "data")
    save_splits(train_ds, test_ds, output_dir=data_dir)

    print("\n==========================================================")
    print("✅ SUCCESS! The actual research dataset is now saved locally.")
    print(f"📁 Training Set (4,500 samples): {os.path.join(data_dir, 'train.jsonl')}")
    print(f"📁 Testing Set (500 samples):   {os.path.join(data_dir, 'test.jsonl')}")
    print("==========================================================")

if __name__ == "__main__":
    main()
