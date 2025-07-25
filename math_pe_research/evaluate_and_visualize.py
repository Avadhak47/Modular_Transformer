import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from models.mathematical_reasoning_model import create_mathematical_reasoning_model

# Utility to load model and tokenizer
def load_model_and_tokenizer(checkpoint_dir):
    pe_config_path = os.path.join(checkpoint_dir, 'pe_config.json')
    with open(pe_config_path, 'r') as f:
        pe_config = json.load(f)
    pe_method = pe_config.get('pe_method', 'rope')
    pe_cfg = pe_config.get('pe_config', {})
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
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

def evaluate_model(model, tokenizer, dataset_name="gsm8k", split="test", max_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    # Load dataset
    dataset = load_dataset(dataset_name, split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    # Metrics
    accuracy = 0
    total = 0
    predictions = []
    references = []
    for example in dataset:
        question = example.get('question', example.get('input', ''))
        answer = example.get('answer', example.get('target', ''))
        inputs = tokenizer(question, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=64)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(answer)
        # Simple accuracy: exact match
        if pred.strip() == answer.strip():
            accuracy += 1
        total += 1
    accuracy = accuracy / total if total > 0 else 0
    # Compute BLEU and ROUGE
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    bleu_score = bleu.compute(predictions=[p.split() for p in predictions], references=[[r.split()] for r in references])["bleu"]
    rouge_score = rouge.compute(predictions=predictions, references=references)
    # Perplexity (optional, for LM tasks)
    # Not always meaningful for math QA, but can be included
    ppl = None
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'generate'):
        # Use negative log likelihood
        losses = []
        for pred, ref in zip(predictions, references):
            inputs = tokenizer(ref, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                losses.append(loss)
        if losses:
            avg_loss = sum(losses) / len(losses)
            ppl = torch.exp(torch.tensor(avg_loss)).item()
    # Print results
    print(f"\nEvaluation Results on {dataset_name} ({split}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  BLEU: {bleu_score:.4f}")
    print(f"  ROUGE-L: {rouge_score['rougeL'].mid.fmeasure:.4f}")
    if ppl:
        print(f"  Perplexity: {ppl:.2f}")
    # Visualization
    metrics = [accuracy, bleu_score, rouge_score['rougeL'].mid.fmeasure]
    metric_names = ["Accuracy", "BLEU", "ROUGE-L"]
    plt.figure(figsize=(7,4))
    plt.bar(metric_names, metrics, color=["#4caf50", "#2196f3", "#ff9800"])
    plt.ylim(0, 1)
    plt.title(f"Model Evaluation on {dataset_name} ({split})")
    plt.ylabel("Score")
    plt.show()
    return {"accuracy": accuracy, "bleu": bleu_score, "rougeL": rouge_score['rougeL'].mid.fmeasure, "perplexity": ppl}

if __name__ == "__main__":
    checkpoint_dir = "./checkpoints/final_model"  # Update as needed
    model, tokenizer, pe_method = load_model_and_tokenizer(checkpoint_dir)
    print(f"\nEvaluating model with {pe_method} PE...")
    results = evaluate_model(model, tokenizer, dataset_name="gsm8k", split="test", max_samples=100)
    print("\n✅ Evaluation and visualization complete!") 