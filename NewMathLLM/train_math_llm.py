import argparse, os
from pathlib import Path

import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import (AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)

from src.modeling_patch import load_deepseek_with_pe

os.environ["WANDB_PROJECT"] = "new-math-llm"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pe", type=str, required=True, choices=["xpos", "sinusoidal", "alibi+"], help="Positional encoding")
    parser.add_argument("--base", type=str, default="deepseek-ai/deepseek-math-7b-instruct", help="HF model repo or local path")
    parser.add_argument("--output", type=str, default="outputs", help="Output dir root")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--context_length", type=int, default=2048)
    return parser.parse_args()


def get_dataset(context_len):
    ds_math = load_dataset("math_dataset", "math_qa", split="train")  # HF community
    ds_gsm8k = load_dataset("gsm8k", "main", split="train")
    ds_dm = load_dataset("deepmind_math", split="train[:5000]")
    dataset = concatenate_datasets([ds_math, ds_gsm8k, ds_dm]).shuffle(seed=42)

    def preprocess(example):
        q = example["question"] if "question" in example else example["problem"]
        a = example["answer"] if "answer" in example else example["solution"]
        text = f"### Problem\n{q}\n### Solution\n{a}"
        return {"text": text}

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    return dataset


def main():
    args = parse_args()
    model = load_deepseek_with_pe(args.base, args.pe)
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token set

    dataset = get_dataset(args.context_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=Path(args.output) / f"{args.pe}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        fp16=True,
        evaluation_strategy="no",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        report_to=["wandb"],
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(Path(args.output) / f"{args.pe}_final")


if __name__ == "__main__":
    main()