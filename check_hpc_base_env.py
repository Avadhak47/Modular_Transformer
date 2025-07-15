import importlib
import sys
import os

REQUIRED_PACKAGES = [
    "torch", "numpy", "scipy", "pandas", "matplotlib", "seaborn", "tqdm", "scikit-learn",
    # Optionally add these if your project uses them and they might be available as modules:
    "transformers", "datasets", "tensorboard", "wandb"
]

PROJECT_MODULES = [
    "src.model",
    "src.config",
    "src.layers.attention",
    "src.layers.encoder",
    "src.layers.decoder",
    "src.layers.embedding",
    "src.layers.feed_forward",
    "src.layers.layer_norm",
    "src.positional_encoding.sinusoidal",
    "src.positional_encoding.rope",
    "src.positional_encoding.alibi",
    "src.positional_encoding.diet",
    "src.positional_encoding.nope",
    "src.positional_encoding.t5_relative",
    "src.utils.mask_utils",
    "src.utils.metrics",
    "src.utils.training_utils",
    "evaluation.mathematical_metrics",
    "data.math_dataset_loader",
    "training.mathematical_reasoning_trainer"
]

missing_packages = []
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing_packages.append(pkg)

missing_modules = []
for mod in PROJECT_MODULES:
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing_modules.append((mod, str(e)))

print("\n=== HPC Base Environment Check ===\n")
if not missing_packages:
    print("All required Python packages are available in the base environment.")
else:
    print("Missing Python packages:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print("\nTry to load a module for the missing package(s) using 'module avail' and 'module load ...'. If not available, request admin install.")

if not missing_modules:
    print("\nAll project modules can be imported successfully.")
else:
    print("\nProject modules with import errors:")
    for mod, err in missing_modules:
        print(f"  - {mod}: {err}")
    print("\nCheck if the missing package(s) above are the cause, or if there are code issues.")

if not missing_packages and not missing_modules:
    print("\n✅ Base environment is sufficient to run the project!")
else:
    print("\n❌ Base environment is missing required packages or has import errors. See above for details.") 