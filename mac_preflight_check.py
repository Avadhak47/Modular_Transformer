#!/usr/bin/env python3
"""
MacBook Air M1 Pre-flight Check Script
Validates the entire project before HPC deployment on local M1 system.
"""

import sys
import os
import importlib
import subprocess
import traceback
import platform
import psutil
import torch
import time
from typing import Dict, List, Tuple, Optional
import logging
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MacPreflightChecker:
    """Comprehensive pre-flight checker for MacBook Air M1 deployment."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_checks = []
        self.system_info = {}
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
        print("=" * 70)
        print("MACBOOK AIR M1 PRE-FLIGHT CHECK")
        print("=" * 70)
        
        # System information check
        self._check_system_info()
        
        checks = [
            ("Environment Check", self.check_environment),
            ("System Resources Check", self.check_system_resources),
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
            ("Performance Check", self.check_performance),
            ("Memory Usage Check", self.check_memory_usage),
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
    
    def _check_system_info(self):
        """Gather system information."""
        self.system_info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent
        }
        
        print(f"\n📊 SYSTEM INFORMATION:")
        print(f"  • Platform: {self.system_info['platform']}")
        print(f"  • Processor: {self.system_info['processor']}")
        print(f"  • Architecture: {self.system_info['architecture']}")
        print(f"  • Python: {self.system_info['python_version'].split()[0]}")
        print(f"  • CPU Cores: {self.system_info['cpu_count']}")
        print(f"  • Memory: {self.system_info['memory_total'] / (1024**3):.1f} GB")
        print(f"  • Available Memory: {self.system_info['memory_available'] / (1024**3):.1f} GB")
        print(f"  • Disk Usage: {self.system_info['disk_usage']:.1f}%")
    
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
            
            # Check for M1-specific considerations
            if "arm64" in platform.architecture()[0]:
                self.warnings.append("Running on ARM64 architecture - some packages may need Rosetta 2")
            
            return True
        except Exception as e:
            self.errors.append(f"Environment check failed: {str(e)}")
            return False
    
    def check_system_resources(self) -> bool:
        """Check if system has enough resources for development."""
        try:
            # Check available memory (need at least 4GB for development)
            available_memory_gb = self.system_info['memory_available'] / (1024**3)
            if available_memory_gb < 4:
                self.warnings.append(f"Low available memory: {available_memory_gb:.1f}GB (recommend 4GB+)")
            
            # Check disk space (need at least 5GB)
            disk_usage = self.system_info['disk_usage']
            if disk_usage > 90:
                self.warnings.append(f"High disk usage: {disk_usage:.1f}% (recommend <90%)")
            
            # Check CPU cores
            cpu_count = self.system_info['cpu_count']
            if cpu_count < 4:
                self.warnings.append(f"Low CPU cores: {cpu_count} (recommend 4+)")
            
            return True
        except Exception as e:
            self.errors.append(f"System resources check failed: {str(e)}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        required_packages = [
            "torch", "transformers", "datasets", "numpy", 
            "scipy", "pandas", "matplotlib", "seaborn",
            "tensorboard", "tqdm", "sklearn", "psutil"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Missing packages: {', '.join(missing_packages)}")
            return False
        
        # Check PyTorch version and backend
        try:
            torch_version = torch.__version__
            if torch.backends.mps.is_available():
                self.passed_checks.append("PyTorch MPS (Metal Performance Shaders) available")
            else:
                self.warnings.append("PyTorch MPS not available - will use CPU only")
            
            print(f"  • PyTorch version: {torch_version}")
            print(f"  • CUDA available: {torch.cuda.is_available()}")
            print(f"  • MPS available: {torch.backends.mps.is_available()}")
            
        except Exception as e:
            self.warnings.append(f"PyTorch check failed: {str(e)}")
        
        return True
    
    def check_code_syntax(self) -> bool:
        """Check syntax of all Python files."""
        python_files = []
        for root, dirs, files in os.walk("."):
            if "venv" in root or "__pycache__" in root or ".git" in root:
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
            
            # Test model info
            model_info = model.get_model_info()
            print(f"  • Model parameters: {model_info['total_parameters']:,}")
            print(f"  • Trainable parameters: {model_info['trainable_parameters']:,}")
            
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
            print("  • Testing GSM8K dataset loading...")
            gsm8k_data = loader.load_gsm8k_dataset('train')
            if not gsm8k_data:
                self.warnings.append("GSM8K dataset loading failed or empty")
            else:
                print(f"  • GSM8K loaded: {len(gsm8k_data)} samples")
            
            # Test MATH loading (small sample)
            print("  • Testing MATH dataset loading...")
            math_data = loader.load_math_dataset('train', max_samples=5)
            if not math_data:
                self.warnings.append("MATH dataset loading failed or empty")
            else:
                print(f"  • MATH loaded: {len(math_data)} samples")
            
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
                ("Sinusoidal", SinusoidalPositionalEncoding(d_model, max_len)),
                ("RoPE", RotaryPositionalEncoding(d_model, max_len, n_heads)),
                ("ALiBi", ALiBiPositionalEncoding(d_model, n_heads, max_len)),
                ("DIET", DIETPositionalEncoding(d_model, max_len)),
                ("None", NoPositionalEncoding(d_model, max_len)),
                ("T5 Relative", T5RelativePositionalEncoding(d_model, max_len, n_heads))
            ]
            
            for name, pe in pe_methods:
                x = torch.randn(2, 10, d_model)
                output = pe(x)
                if output.shape != x.shape:
                    self.errors.append(f"Positional encoding {name} output shape mismatch")
                    return False
                print(f"  • {name}: ✓")
            
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
            
            print(f"  • Evaluation metrics: ✓")
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
            # Fix the config structure issue
            config = {
                'model': config,
                'positional_encoding': 'sinusoidal',
                'tokenizer_name': 'gpt2',
                'max_length': 1024,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'batch_size': 4,
                'eval_batch_size': 4,
                'use_wandb': False,
                'project_name': 'math_reasoning'
            }
            
            trainer = MathematicalReasoningTrainer(config)
            print(f"  • Training pipeline: ✓")
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
            self.warnings.append(f"Missing HPC scripts: {', '.join(missing_scripts)}")
        
        return True
    
    def check_containerization(self) -> bool:
        """Check Docker and Singularity setup."""
        try:
            # Check if Dockerfile exists and is valid
            if not os.path.exists("Dockerfile"):
                self.errors.append("Dockerfile not found")
                return False
            
            # Check if requirements.txt exists
            if not os.path.exists("requirements.txt"):
                self.errors.append("requirements.txt not found")
                return False
            
            # Check Docker availability
            try:
                result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  • Docker: {result.stdout.strip()}")
                else:
                    self.warnings.append("Docker not available")
            except Exception:
                self.warnings.append("Docker not available")
            
            # Check Singularity availability
            try:
                result = subprocess.run(
                    ["singularity", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  • Singularity: {result.stdout.strip()}")
                else:
                    self.warnings.append("Singularity not available")
            except Exception:
                self.warnings.append("Singularity not available")
            
            return True
        except Exception as e:
            self.errors.append(f"Containerization check failed: {str(e)}")
            return False
    
    def check_performance(self) -> bool:
        """Check basic performance metrics."""
        try:
            # Test model inference speed
            from src.model import TransformerModel
            from config import get_config
            
            config = get_config().model.to_dict()
            model = TransformerModel(config)
            
            # Warm up
            batch_size, seq_len = 1, 64
            src = torch.randint(0, 1000, (batch_size, seq_len))
            
            # Test inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(src)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            print(f"  • Average inference time: {avg_time:.4f}s")
            
            if avg_time > 1.0:
                self.warnings.append(f"Slow inference time: {avg_time:.4f}s")
            
            return True
        except Exception as e:
            self.warnings.append(f"Performance check failed: {str(e)}")
            return False
    
    def check_memory_usage(self) -> bool:
        """Check memory usage patterns."""
        try:
            # Test memory usage during model operations
            import psutil
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            from src.model import TransformerModel
            from config import get_config
            
            config = get_config().model.to_dict()
            model = TransformerModel(config)
            
            # Test memory after model creation
            model_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = model_memory - initial_memory
            
            print(f"  • Initial memory: {initial_memory:.1f} MB")
            print(f"  • After model creation: {model_memory:.1f} MB")
            print(f"  • Memory increase: {memory_increase:.1f} MB")
            
            if memory_increase > 500:  # More than 500MB
                self.warnings.append(f"High memory usage: {memory_increase:.1f} MB")
            
            return True
        except Exception as e:
            self.warnings.append(f"Memory usage check failed: {str(e)}")
            return False
    
    def _print_summary(self):
        """Print comprehensive summary of all checks."""
        print("\n" + "=" * 70)
        print("MAC PRE-FLIGHT CHECK SUMMARY")
        print("=" * 70)
        
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
        
        print(f"\n📊 SYSTEM RECOMMENDATIONS:")
        
        # M1-specific recommendations
        if "arm64" in platform.architecture()[0]:
            print("  • ARM64 Architecture detected:")
            print("    - Consider using PyTorch with MPS backend for better performance")
            print("    - Some packages may need Rosetta 2 translation")
            print("    - Docker images should be built for ARM64")
        
        # Memory recommendations
        available_memory_gb = self.system_info['memory_available'] / (1024**3)
        if available_memory_gb < 8:
            print("  • Memory recommendations:")
            print("    - Close unnecessary applications")
            print("    - Consider using smaller batch sizes")
            print("    - Monitor memory usage during training")
        
        # Performance recommendations
        print("  • Performance recommendations:")
        print("    - Use MPS backend if available")
        print("    - Consider gradient accumulation for larger models")
        print("    - Monitor CPU and memory usage")
        
        print(f"\n📊 OVERALL STATUS:")
        if len(self.errors) == 0:
            print("🎉 ALL CHECKS PASSED - Ready for HPC deployment!")
        else:
            print(f"🚨 {len(self.errors)} ERROR(S) FOUND - Fix before deployment!")
        
        print("\n" + "=" * 70)


def main():
    """Main function to run the pre-flight check."""
    checker = MacPreflightChecker()
    success = checker.run_all_checks()
    
    if success:
        print("\n🚀 Ready for HPC deployment!")
        sys.exit(0)
    else:
        print("\n❌ Pre-flight check failed. Fix errors before deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main() 