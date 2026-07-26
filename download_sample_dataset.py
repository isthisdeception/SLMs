"""
Helper script to download a small local sample (e.g. 10 examples) of the dataset 
so you can inspect its raw structure, conversations, and reasoning traces on your local computer.
"""

import json
import os
from datasets import load_dataset

def main():
    print("Downloading a 10-sample inspection preview of HuggingFaceH4/Bespoke-Stratos-17k...")
    
    # Download top 10 rows
    ds = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train[:10]")
    
    os.makedirs("data", exist_ok=True)
    sample_path = os.path.join("data", "sample_inspection.json")
    
    samples = [item for item in ds]
    
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Downloaded 10 sample records to '{sample_path}'!")
    print("You can now open 'data/sample_inspection.json' in VSCode to examine the exact prompts and reasoning traces!")

if __name__ == "__main__":
    main()
