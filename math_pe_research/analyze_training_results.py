import os
import json
from pathlib import Path

def analyze_results(results_dir, checkpoint_dir):
    # Find results file
    results_file = os.path.join(results_dir, 'final_results.json')
    if not os.path.exists(results_file):
        print(f"❌ No results file found in {results_dir}")
        return
    with open(results_file, 'r') as f:
        results = json.load(f)
    print("\n📈 Results from final_results.json:")
    for k, v in results.items():
        print(f"   {k}: {v}")

    # Find pe_config.json
    pe_config_path = os.path.join(checkpoint_dir, 'pe_config.json')
    if not os.path.exists(pe_config_path):
        print(f"❌ No pe_config.json found in {checkpoint_dir}")
        return
    with open(pe_config_path, 'r') as f:
        pe_config = json.load(f)
    pe_method = pe_config.get('pe_method', 'unknown')
    pe_cfg = pe_config.get('pe_config', {})
    print("\n🎯 MODEL SUMMARY:")
    print(f"   PE Method: {pe_method}")
    print(f"   Base Model: {results.get('model_size', 'unknown')}")
    print(f"   Training Steps: {results.get('total_steps', 'unknown')}")
    print(f"   LoRA Enabled: {results.get('use_lora', True)}")
    print(f"   PE Config: {pe_cfg}")

    # Check for model weights
    weights_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.bin') or f.endswith('.safetensors')]
    if not weights_files:
        print("❌ No model weights (.bin or .safetensors) found in checkpoint directory!")
    else:
        print(f"\n💾 Found {len(weights_files)} model weights files:")
        for wf in weights_files:
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, wf)) / (1024*1024)
            print(f"   📄 {wf} ({size_mb:.1f} MB)")
            if size_mb < 1.0:
                print(f"   ⚠️  Warning: {wf} is very small! Model may not have been saved correctly.")

if __name__ == "__main__":
    results_dir = "./results"  # Update as needed
    checkpoint_dir = "./checkpoints/final_model"  # Update as needed
    print("\n📊 TRAINING RESULTS ANALYSIS\n" + "="*50)
    analyze_results(results_dir, checkpoint_dir)
    print("\n✅ Analysis complete!\n") 