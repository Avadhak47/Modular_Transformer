import os
import json
from datasets import load_dataset

os.makedirs("local_data", exist_ok=True)

# Download GSM8K
print("Downloading GSM8K...")
gsm8k = load_dataset("openai/gsm8k", "main")
for split in ["train", "test"]:
    with open(f"local_data/gsm8k_{split}.jsonl", "w") as f:
        for item in gsm8k[split]:
            f.write(json.dumps(item) + "\n")

# Download MATH
print("Downloading MATH train...")
math = load_dataset("Dahoas/MATH", split="train")
with open("local_data/math_train.jsonl", "w") as f:
    for item in math:
        f.write(json.dumps(item) + "\n")

print("Downloading MATH test...")
math_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
with open("local_data/math_test.jsonl", "w") as f:
    for item in math_test:
        f.write(json.dumps(item) + "\n")

print("Datasets downloaded and saved to local_data/") 