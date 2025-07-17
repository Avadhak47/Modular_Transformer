# NewMathLLM – Mathematical-Reasoning Language-Model Toolkit

This repo is a **green-field rewrite** that bootstraps state-of-the-art math-centric large-language-models on the IIT-Delhi PADUM cluster.  It re-uses only the positional-encoding (PE) ideas from the earlier Transformer project; everything else is rebuilt around latest open-source checkpoints.

Key components
--------------
1. **Base model**: [`DeepSeekMath-Instruct-7B`](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct) (and the RLHF sibling).
2. **PE variants** (under `src/pe/`):
   • `xpos.py` – improved RoPE with extrapolation (Sun et al., 2023)
   • `alibi_plus.py` – bias-based PE with learnable slope offset
   • `sinusoidal.py` – classic baseline (ported from the old repo)
3. **Model patcher** (`src/modeling_patch.py`) – injects any PE implementation into a pretrained LLaMA/LLaMA-like checkpoint **without retraining the rest of the layers from scratch**.
4. **Dataset**:  a composite of
   • MATH (12.5 K problems)
   • GSM8K
   • [`DeepMind Mathematics`](https://github.com/deepmind/mathematics_dataset)
   • `LatexFormulaMath` subset from Proof-Pile

   All fetched via the 🤗 `datasets` hub (`scripts/download_datasets.py`).
5. **Training script** (`train_math_llm.py`):
   • Uses HuggingFace `Trainer` + mixed-precision + gradient-accum.
6. **Cluster automation** (`hpc/`):
   • `submit_multi_node_deepseek.pbs` – 5-node PBS job, one node per PE.
   • `experiment_launcher.py` – CLI entrypoint to launch/re-launch a PE run; handles resume/best-checkpoint logic per node.

Quickstart
----------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_datasets.py  # ~8 GB
python train_math_llm.py --pe xpos --base deepseek-ai/deepseek-math-7b-instruct
```

For full PADUM automation, see `hpc/HPC_DEPLOYMENT_GUIDE.md`.