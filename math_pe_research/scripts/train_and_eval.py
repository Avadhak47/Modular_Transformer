import argparse
import os
import json
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import wandb

from models.mathematical_reasoning_model import create_mathematical_reasoning_model
from data.math_dataset_loader import MathDatasetLoader

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a mathematical reasoning model.")
    parser.add_argument('--model_size', type=str, default="deepseek-ai/deepseek-math-7b-instruct",
                        help="Base model size/name (e.g., deepseek-ai/deepseek-math-7b-instruct, deepseek-ai/deepseek-math-1.3b-instruct, etc.)")
    parser.add_argument('--pe', type=str, default="rope", choices=["rope", "alibi", "sinusoidal", "diet", "t5_relative", "math_adaptive"],
                        help="Positional encoding method")
    parser.add_argument('--experiment_name', type=str, required=True, help="Experiment name")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument('--result_dir', type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument('--max_steps', type=int, default=1000, help="Max training steps")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--max_length', type=int, default=4096, help="Max sequence length")
    parser.add_argument('--datasets', type=str, default="gsm8k,math", help="Comma-separated list of datasets")
    parser.add_argument('--wandb_project', type=str, default="math_pe_research", help="wandb project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="wandb entity (optional)")
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cache downloaded models and datasets')
    parser.add_argument('--load_in_4bit', action='store_true', help='Enable 4-bit quantization (requires bitsandbytes, not supported on Kaggle by default)')
    args = parser.parse_args()

    # Setup directories
    checkpoint_dir = Path(args.checkpoint_dir)
    result_dir = Path(args.result_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = checkpoint_dir / "data_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

    # wandb setup
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.experiment_name,
        config=vars(args)
    )

    # Model and tokenizer
    model = create_mathematical_reasoning_model(
        pe_method=args.pe,
        base_model=args.model_size,
        load_in_4bit=args.load_in_4bit,
        use_lora=True,
        cache_dir=args.cache_dir
    )
    tokenizer = model.tokenizer

    # Data loading
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    loader = MathDatasetLoader(tokenizer=tokenizer, max_length=args.max_length, cache_dir=str(cache_dir))
    train_problems = loader.load_multiple_datasets(dataset_names, split='train', max_samples_per_dataset=10000)
    eval_problems = loader.load_multiple_datasets(dataset_names, split='test', max_samples_per_dataset=1000)
    train_dataset = loader.create_pytorch_dataset(train_problems, is_training=True)
    eval_dataset = loader.create_pytorch_dataset(eval_problems, is_training=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        logging_dir=str(checkpoint_dir / "logs"),
        num_train_epochs=4,  # Use max_steps instead
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=50,
        report_to="wandb",
        fp16=True,
        remove_unused_columns=False,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        weight_decay=0.01,
    )

    # Trainer
    trainer = Trainer(
        model=model.base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Training
    print("Starting training...")
    train_result = trainer.train()
    print("Training complete.")

    # Save model and tokenizer
    print(f"Saving model and tokenizer to {checkpoint_dir} ...")
    model.save_pretrained(str(checkpoint_dir))

    # Evaluation
    print("Running evaluation...")
    eval_result = trainer.evaluate()
    print("Evaluation complete.")

    # Save evaluation results
    results = {
        'experiment_name': args.experiment_name,
        'pe_method': args.pe,
        'model_size': args.model_size,
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result.get('eval_loss', None),
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
        'total_steps': train_result.global_step
    }
    results_file = result_dir / "final_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # wandb log and finish
    wandb.log(results)
    wandb.finish()

if __name__ == "__main__":
    main() 