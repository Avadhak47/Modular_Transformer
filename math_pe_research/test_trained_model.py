import os
import json
from pathlib import Path
from models.mathematical_reasoning_model import create_mathematical_reasoning_model
from transformers import AutoTokenizer
import torch

def load_model_and_tokenizer(checkpoint_dir):
    # Load PE config
    pe_config_path = os.path.join(checkpoint_dir, 'pe_config.json')
    if not os.path.exists(pe_config_path):
        raise FileNotFoundError(f"pe_config.json not found in {checkpoint_dir}")
    with open(pe_config_path, 'r') as f:
        pe_config = json.load(f)
    pe_method = pe_config.get('pe_method', 'rope')
    pe_cfg = pe_config.get('pe_config', {})

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # Load model
    model = create_mathematical_reasoning_model(
        pe_method=pe_method,
        pe_config=pe_cfg,
        base_model=checkpoint_dir,
        use_lora=True,
        load_in_4bit=False,
        enable_gradient_checkpointing=False,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    return model, tokenizer, pe_method

if __name__ == "__main__":
    checkpoint_dir = "./checkpoints/final_model"  # Update as needed
    print("\n📥 Loading trained model...")
    model, tokenizer, pe_method = load_model_and_tokenizer(checkpoint_dir)
    print(f"✅ Model loaded with {pe_method} PE")

    # Example inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    problems = [
        "What is 15 + 27?",
        "If a rectangle has length 8 and width 5, what is its area?",
        "Solve for x: 2x + 5 = 13",
        "What is the square root of 144?",
        "A train travels 120 miles in 2 hours. What is its speed?"
    ]
    for i, problem in enumerate(problems, 1):
        inputs = tokenizer(problem, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=64)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n{i}. Problem: {problem}\n   Solution: {answer}") 