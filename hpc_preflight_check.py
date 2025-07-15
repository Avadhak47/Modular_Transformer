#!/usr/bin/env python3
"""
HPC Pre-flight Check Script
Validates the entire project before HPC deployment.
"""

import sys
import os
import importlib
import subprocess
import traceback
from typing import Dict, List, Tuple, Optional
import logging
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class HPCPreflightChecker:
    """Comprehensive pre-flight checker for HPC deployment."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the checker."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run_all_checks(self) -> bool:
        """Run all pre-flight checks and return success status."""
        print("=" * 60)
        print("HPC PRE-FLIGHT CHECK")
        print("=" * 60)
        
        checks = [
            ("Environment Check", self.check_environment),
            ("Dependencies Check", self.check_dependencies),
            ("Code Syntax Check", self.check_code_syntax),
            ("Import Check", self.check_imports),
            ("Model Initialization Check", self.check_model_init),
            ("Data Loading Check", self.check_data_loading),
            ("Positional Encoding Check", self.check_positional_encodings),
            ("Evaluation Metrics Check", self.check_evaluation_metrics),
            ("Training Pipeline Check", self.check_training_pipeline),
            ("HPC Scripts Check", self.check_hpc_scripts),
            ("Docker/Singularity Check", self.check_containerization),
        ]
        
        for check_name, check_func in checks:
            print(f"\n🔍 {check_name}...")
            try:
                result = check_func()
                if result:
                    self.passed_checks.append(check_name)
                    print(f"✅ {check_name} - PASSED")
                else:
                    print(f"❌ {check_name} - FAILED")
            except Exception as e:
                self.errors.append(f"{check_name}: {str(e)}")
                print(f"❌ {check_name} - ERROR: {str(e)}")
        
        self._print_summary()
        return len(self.errors) == 0
    
    def check_environment(self) -> bool:
        """Check Python environment and basic setup."""
        try:
            # Check Python version
            version = sys.version_info
            if version < (3, 8):
                self.errors.append(f"Python version {version.major}.{version.minor} is too old. Need 3.8+")
                return False
            
            # Check if we're in the right directory
            if not os.path.exists("src/model.py"):
                self.errors.append("Not in project root directory")
                return False
            
            # Check for required directories
            required_dirs = ["src", "data", "evaluation", "training"]
            for dir_name in required_dirs:
                if not os.path.exists(dir_name):
                    self.errors.append(f"Missing directory: {dir_name}")
                    return False
            
            return True
        except Exception as e:
            self.errors.append(f"Environment check failed: {str(e)}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available in the current environment (module-based)."""
        # List only packages that should be available via modules or base conda env
        required_packages = [
            "torch", "numpy", "scipy", "pandas", "matplotlib", "seaborn",
            "tqdm", "scikit-learn"
        ]
        # Optionally add: 'transformers', 'datasets', 'tensorboard', 'wandb' if available as modules
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        if missing_packages:
            self.errors.append(
                f"Missing packages: {', '.join(missing_packages)}. "
                "Check if a module is available for these packages using 'module avail' and load it if possible. "
                "If not, contact HPC support to request installation."
            )
            return False
        return True
    
    def check_code_syntax(self) -> bool:
        """Check syntax of all Python files."""
        python_files = []
        for root, dirs, files in os.walk("."):
            if "venv" in root or "__pycache__" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        
        syntax_errors = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    compile(f.read(), file_path, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                syntax_errors.append(f"{file_path}: {str(e)}")
        
        if syntax_errors:
            self.errors.append(f"Syntax errors found:\n" + "\n".join(syntax_errors))
            return False
        
        return True
    
    def check_imports(self) -> bool:
        """Check if all modules can be imported."""
        modules_to_test = [
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
        
        import_errors = []
        for module in modules_to_test:
            try:
                importlib.import_module(module)
            except Exception as e:
                import_errors.append(f"{module}: {str(e)}")
        
        if import_errors:
            self.errors.append(f"Import errors:\n" + "\n".join(import_errors))
            return False
        
        return True
    
    def check_model_init(self) -> bool:
        """Check if the model can be initialized."""
        try:
            from src.model import TransformerModel
            from config import get_config
            
            config = get_config().model.to_dict()
            model = TransformerModel(config)
            
            # Test forward pass with dummy data
            batch_size, seq_len = 2, 10
            src = torch.randint(0, 1000, (batch_size, seq_len))
            tgt = torch.randint(0, 1000, (batch_size, seq_len))
            
            output = model(src, tgt)
            if output.shape != (batch_size, seq_len, config['vocab_size']):
                self.errors.append("Model output shape mismatch")
                return False
            
            return True
        except Exception as e:
            self.errors.append(f"Model initialization failed: {str(e)}")
            return False
    
    def check_data_loading(self) -> bool:
        """Check if datasets can be loaded."""
        try:
            from data.math_dataset_loader import MathematicalDatasetLoader
            
            loader = MathematicalDatasetLoader()
            
            # Test GSM8K loading (small sample)
            gsm8k_data = loader.load_gsm8k_dataset('train')
            if not gsm8k_data:
                self.warnings.append("GSM8K dataset loading failed or empty")
            
            # Test MATH loading (small sample)
            math_data = loader.load_math_dataset('train', max_samples=5)
            if not math_data:
                self.warnings.append("MATH dataset loading failed or empty")
            
            return True
        except Exception as e:
            self.errors.append(f"Data loading failed: {str(e)}")
            return False
    
    def check_positional_encodings(self) -> bool:
        """Check if all positional encodings work."""
        try:
            from src.positional_encoding.sinusoidal import SinusoidalPositionalEncoding
            from src.positional_encoding.rope import RotaryPositionalEncoding
            from src.positional_encoding.alibi import ALiBiPositionalEncoding
            from src.positional_encoding.diet import DIETPositionalEncoding
            from src.positional_encoding.nope import NoPositionalEncoding
            from src.positional_encoding.t5_relative import T5RelativePositionalEncoding
            
            d_model, max_len = 128, 512
            n_heads = 8
            
            # Test each positional encoding
            pe_methods = [
                SinusoidalPositionalEncoding(d_model, max_len),
                RotaryPositionalEncoding(d_model, max_len, n_heads),
                ALiBiPositionalEncoding(d_model, n_heads, max_len),
                DIETPositionalEncoding(d_model, max_len),
                NoPositionalEncoding(d_model, max_len),
                T5RelativePositionalEncoding(d_model, max_len, n_heads)
            ]
            
            for pe in pe_methods:
                x = torch.randn(2, 10, d_model)
                output = pe(x)
                if output.shape != x.shape:
                    self.errors.append(f"Positional encoding {pe.__class__.__name__} output shape mismatch")
                    return False
            
            return True
        except Exception as e:
            self.errors.append(f"Positional encoding check failed: {str(e)}")
            return False
    
    def check_evaluation_metrics(self) -> bool:
        """Check if evaluation metrics work."""
        try:
            from evaluation.mathematical_metrics import MathematicalReasoningEvaluator
            
            evaluator = MathematicalReasoningEvaluator()
            
            # Test with dummy data
            predictions = ["The answer is 42", "x = 15"]
            references = ["42", "15"]
            
            metrics = evaluator.exact_match_accuracy(predictions, references)
            if not isinstance(metrics, dict):
                self.errors.append("Evaluation metrics return type error")
                return False
            
            return True
        except Exception as e:
            self.errors.append(f"Evaluation metrics check failed: {str(e)}")
            return False
    
    def check_training_pipeline(self) -> bool:
        """Check if training pipeline components work."""
        try:
            from training.mathematical_reasoning_trainer import MathematicalReasoningTrainer
            from config import get_config
            
            config = get_config().model.to_dict()
            trainer = MathematicalReasoningTrainer(config)
            
            return True
        except Exception as e:
            self.errors.append(f"Training pipeline check failed: {str(e)}")
            return False
    
    def check_hpc_scripts(self) -> bool:
        """Check if HPC scripts exist and are valid."""
        required_scripts = [
            "submit_training.pbs",
            "Singularity.def",
            "setup_padum_automation.sh",
            "export_and_transfer.sh",
            "monitor_padum_job.sh"
        ]
        
        missing_scripts = []
        for script in required_scripts:
            if not os.path.exists(script):
                missing_scripts.append(script)
        
        if missing_scripts:
            self.errors.append(f"Missing HPC scripts: {', '.join(missing_scripts)}")
            return False
        
        return True
    
    def check_containerization(self) -> bool:
        """Skip containerization check (no longer used)."""
        self.warnings.append("Containerization (Docker/Singularity) is not used in the current workflow.")
        return True
    
    def _print_summary(self):
        """Print comprehensive summary of all checks."""
        print("\n" + "=" * 60)
        print("PRE-FLIGHT CHECK SUMMARY")
        print("=" * 60)
        
        print(f"\n✅ PASSED CHECKS ({len(self.passed_checks)}):")
        for check in self.passed_checks:
            print(f"  • {check}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        print(f"\n📊 OVERALL STATUS:")
        if len(self.errors) == 0:
            print("🎉 ALL CHECKS PASSED - Ready for HPC deployment!")
        else:
            print(f"🚨 {len(self.errors)} ERROR(S) FOUND - Fix before deployment!")
        
        print("\n" + "=" * 60)


def main():
    """Main function to run the pre-flight check."""
    checker = HPCPreflightChecker()
    success = checker.run_all_checks()
    
    if success:
        print("\n🚀 Ready for HPC deployment!")
        sys.exit(0)
    else:
        print("\n❌ Pre-flight check failed. Fix errors before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main() 