import os
import sys

# Ensure current directory and script directory are in sys.path BEFORE importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import yaml
from src.data_utils import (
    setup_environment,
    load_and_filter_dataset,
    prepare_splits,
    save_splits,
    compute_token_stats
)

def main():
    print("=== Executing Phase 1: Environment Setup & Dataset Preparation ===")
    
    # Load config
    config_path = os.path.join(current_dir, "configs", "qlora_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("Configuration loaded successfully.")

    # Setup env
    env_info = setup_environment(seed=config['project']['seed'])

    # Load dataset
    raw_ds = load_and_filter_dataset(
        dataset_name=config['data']['dataset_name'],
        fallback_dataset_name=config['data']['fallback_dataset_name'],
        n_samples=config['data']['n_samples'],
        seed=config['project']['seed']
    )

    # Format & Split
    train_ds, test_ds = prepare_splits(
        dataset=raw_ds,
        train_ratio=config['data']['train_ratio'],
        seed=config['project']['seed'],
        model_type=config['model']['model_type']
    )

    # Inspect sample
    print("\n=== Sample Formatted Entry ===")
    print(train_ds[0]['text'][:500])
    print("...\n")

    # Save
    data_dir = os.path.join(current_dir, "data")
    save_splits(train_ds, test_ds, output_dir=data_dir)
    print("=== Phase 1 Completed Successfully ===")

if __name__ == "__main__":
    main()
