import argparse
import os
from pathlib import Path
import json

from training.mathematical_reasoning_trainer import (
    MathematicalReasoningTrainer,
    get_mathematical_reasoning_config,
)


def main():
    parser = argparse.ArgumentParser(description="HPC launcher for mathematical reasoning trainer")
    parser.add_argument("--positional_encoding", type=str, required=True, choices=[
        "sinusoidal", "rope", "alibi", "diet", "t5_relative", "nope"],
        help="Positional encoding variant to train")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base checkpoint (SOTA) for initialisation")
    parser.add_argument("--checkpoint_root", type=str, default="/scratch/$USER/math_reasoning/checkpoints",
                        help="Root directory for storing checkpoints")
    parser.add_argument("--results_root", type=str, default="/scratch/$USER/math_reasoning/results",
                        help="Root directory for evaluation results")
    parser.add_argument("--experiment_suffix", type=str, default="", help="Optional wandb/run suffix")
    args = parser.parse_args()

    # Resolve env vars in provided paths (e.g., $USER)
    username = os.environ.get("USER", "user")
    checkpoint_root = Path(Path(args.checkpoint_root).expanduser().as_posix().replace("$USER", username))
    results_root = Path(Path(args.results_root).expanduser().as_posix().replace("$USER", username))
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    # Determine resume checkpoint if exists
    best_ckpt_link = checkpoint_root / f"{args.positional_encoding}_best.pt"
    resume_path = str(best_ckpt_link) if best_ckpt_link.exists() else None

    # Build config
    config = get_mathematical_reasoning_config(args.positional_encoding)
    config.update({
        "checkpoint_dir": str(checkpoint_root),
        "results_dir": str(results_root),
        "base_model_path": args.base_model_path,
        "resume_from_checkpoint": resume_path,
        "experiment_suffix": args.experiment_suffix,
    })

    # Persist final config for record keeping
    cfg_file = results_root / f"run_config_{args.positional_encoding}.json"
    with open(cfg_file, "w") as f:
        json.dump(config, f, indent=2)

    trainer = MathematicalReasoningTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()