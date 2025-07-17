#!/usr/bin/env python3
"""
Comprehensive Experiment Simulation for Mathematical Reasoning PE Research

This script simulates the entire experimental pipeline to identify potential issues
before running on the HPC cluster. It tests all components and provides detailed
error analysis and recommendations.

Usage: python simulate_experiment.py [--pe_method rope] [--quick] [--verbose]
"""

import os
import sys
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import psutil
import torch
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Simulation result for a specific component."""
    component: str
    success: bool
    error_message: Optional[str] = None
    warning_messages: List[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []
        if self.recommendations is None:
            self.recommendations = []


class ExperimentSimulator:
    """Comprehensive simulator for the mathematical reasoning experiment."""
    
    def __init__(self, pe_method: str = "rope", quick_mode: bool = False, verbose: bool = False):
        self.pe_method = pe_method
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.results: List[SimulationResult] = []
        self.start_time = time.time()
        
        # Simulation parameters
        self.sim_batch_size = 2 if quick_mode else 4
        self.sim_max_length = 512 if quick_mode else 2048
        self.sim_max_samples = 10 if quick_mode else 100
        self.sim_max_steps = 5 if quick_mode else 50
        
        logger.info(f"Initializing experiment simulation for PE method: {pe_method}")
        logger.info(f"Quick mode: {quick_mode}, Verbose: {verbose}")
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete simulation of the experiment pipeline."""
        
        logger.info("🚀 Starting comprehensive experiment simulation...")
        
        # Test components in order
        test_components = [
            ("Environment Setup", self._test_environment_setup),
            ("Dependencies Check", self._test_dependencies),
            ("Positional Encoding", self._test_positional_encoding),
            ("Model Loading", self._test_model_loading),
            ("Dataset Loading", self._test_dataset_loading),
            ("Data Processing", self._test_data_processing),
            ("Training Setup", self._test_training_setup),
            ("Training Loop", self._test_training_loop),
            ("Model Saving", self._test_model_saving),
            ("Evaluation", self._test_evaluation),
            ("Memory Usage", self._test_memory_usage),
            ("HPC Compatibility", self._test_hpc_compatibility)
        ]
        
        for component_name, test_func in test_components:
            logger.info(f"🧪 Testing: {component_name}")
            result = self._run_component_test(component_name, test_func)
            self.results.append(result)
            
            if not result.success:
                logger.error(f"❌ CRITICAL: {component_name} failed!")
                if not self.quick_mode:
                    logger.info("Stopping simulation due to critical failure")
                    break
            else:
                logger.info(f"✅ {component_name} passed")
        
        # Generate final report
        return self._generate_simulation_report()
    
    def _run_component_test(self, component_name: str, test_func) -> SimulationResult:
        """Run a single component test with error handling."""
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = test_func()
            if result is None:
                result = SimulationResult(component_name, True)
            elif isinstance(result, bool):
                result = SimulationResult(component_name, result)
            elif not isinstance(result, SimulationResult):
                result = SimulationResult(component_name, True)
            
            result.component = component_name
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                error_msg += f"\n{traceback.format_exc()}"
            
            result = SimulationResult(
                component_name,
                False,
                error_message=error_msg
            )
        
        # Calculate metrics
        result.execution_time = time.time() - start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        result.memory_usage = current_memory - initial_memory
        
        return result
    
    def _test_environment_setup(self) -> SimulationResult:
        """Test environment setup and configuration."""
        result = SimulationResult("Environment Setup", True)
        
        # Check Python version
        if sys.version_info < (3, 8):
            result.success = False
            result.error_message = f"Python {sys.version} is too old. Requires Python 3.8+"
            return result
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            result.warning_messages.append("CUDA not available - training will be slow")
            result.recommendations.append("Enable GPU for faster training")
        else:
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Found {gpu_count} GPU(s), Memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 8:
                result.warning_messages.append(f"Low GPU memory: {gpu_memory:.1f}GB")
                result.recommendations.append("Consider using smaller batch sizes or 4-bit quantization")
        
        # Check available system memory
        available_memory = psutil.virtual_memory().available / 1024**3
        if available_memory < 16:
            result.warning_messages.append(f"Low system memory: {available_memory:.1f}GB")
            result.recommendations.append("Monitor memory usage during training")
        
        return result
    
    def _test_dependencies(self) -> SimulationResult:
        """Test all required dependencies."""
        result = SimulationResult("Dependencies", True)
        
        required_packages = [
            ("torch", "2.0.0"),
            ("transformers", "4.30.0"),
            ("datasets", "2.10.0"),
            ("accelerate", "0.20.0"),
            ("peft", "0.4.0"),
            ("wandb", "0.15.0")
        ]
        
        missing_packages = []
        version_issues = []
        
        for package, min_version in required_packages:
            try:
                module = __import__(package)
                if hasattr(module, '__version__'):
                    version = module.__version__
                    if self._compare_versions(version, min_version) < 0:
                        version_issues.append(f"{package} {version} < {min_version}")
                else:
                    result.warning_messages.append(f"Cannot determine {package} version")
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            result.success = False
            result.error_message = f"Missing packages: {', '.join(missing_packages)}"
            result.recommendations.append("Install missing packages with: pip install -r requirements.txt")
        
        if version_issues:
            result.warning_messages.extend(version_issues)
            result.recommendations.append("Update packages to recommended versions")
        
        return result
    
    def _test_positional_encoding(self) -> SimulationResult:
        """Test positional encoding implementations."""
        result = SimulationResult("Positional Encoding", True)
        
        try:
            from positional_encoding import get_positional_encoding, PE_REGISTRY
            
            # Test if PE method exists
            if self.pe_method not in PE_REGISTRY:
                result.success = False
                result.error_message = f"PE method '{self.pe_method}' not found. Available: {list(PE_REGISTRY.keys())}"
                return result
            
            # Test PE instantiation
            pe_layer = get_positional_encoding(
                self.pe_method,
                d_model=512,
                max_seq_len=self.sim_max_length
            )
            
            # Test PE forward pass
            batch_size = self.sim_batch_size
            seq_len = 128
            
            if self.pe_method in ['rope', 'math_adaptive']:
                # Test with attention-like input
                x = torch.randn(batch_size, seq_len, 8, 64)  # (B, L, H, D)
                token_ids = torch.randint(0, 1000, (batch_size, seq_len))
                output = pe_layer(x, token_ids=token_ids)
            else:
                # Test with embedding-like input
                x = torch.randn(batch_size, seq_len, 512)
                output = pe_layer(x)
            
            if output.shape != x.shape:
                result.warning_messages.append(f"PE output shape mismatch: {output.shape} vs {x.shape}")
            
            logger.info(f"PE {self.pe_method} test passed - Input: {x.shape}, Output: {output.shape}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.recommendations.append("Check PE implementation for the specified method")
        
        return result
    
    def _test_model_loading(self) -> SimulationResult:
        """Test model loading and configuration."""
        result = SimulationResult("Model Loading", True)
        
        try:
            from models.mathematical_reasoning_model import create_mathematical_reasoning_model
            
            # Test with smaller model for simulation
            test_model_name = "microsoft/DialoGPT-small"  # Much smaller for testing
            
            logger.info(f"Testing model creation with {test_model_name}")
            
            model = create_mathematical_reasoning_model(
                pe_method=self.pe_method,
                base_model=test_model_name,
                use_lora=True,
                load_in_4bit=False,  # Disable for simulation
                device_map=None  # Let it use default device
            )
            
            # Test model forward pass
            tokenizer = model.tokenizer
            test_input = "What is 2 + 2?"
            inputs = tokenizer(test_input, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logger.info(f"Model forward pass successful - Loss: {outputs.loss}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if "out of memory" in str(e).lower():
                result.recommendations.append("Try smaller model or enable 4-bit quantization")
            elif "not found" in str(e).lower():
                result.recommendations.append("Check model name and HuggingFace access")
        
        return result
    
    def _test_dataset_loading(self) -> SimulationResult:
        """Test dataset loading and processing."""
        result = SimulationResult("Dataset Loading", True)
        
        try:
            from data.math_dataset_loader import MathDatasetLoader, create_demo_dataset
            
            # Use demo dataset for simulation
            demo_problems = create_demo_dataset()
            
            if len(demo_problems) == 0:
                result.warning_messages.append("Demo dataset is empty")
            
            logger.info(f"Demo dataset loaded: {len(demo_problems)} problems")
            
            # Test with actual dataset loader
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            loader = MathDatasetLoader(
                tokenizer=tokenizer,
                max_length=self.sim_max_length
            )
            
            # Test PyTorch dataset creation
            dataset = loader.create_pytorch_dataset(demo_problems, is_training=True)
            
            if len(dataset) != len(demo_problems):
                result.warning_messages.append("Dataset size mismatch after processing")
            
            # Test data loading
            sample = dataset[0]
            required_keys = ['input_ids', 'attention_mask', 'labels']
            missing_keys = [key for key in required_keys if key not in sample]
            
            if missing_keys:
                result.warning_messages.append(f"Missing keys in dataset sample: {missing_keys}")
            
            logger.info(f"Dataset sample keys: {list(sample.keys())}")
            logger.info(f"Sample input_ids shape: {sample['input_ids'].shape}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.recommendations.append("Check dataset loading and processing pipeline")
        
        return result
    
    def _test_data_processing(self) -> SimulationResult:
        """Test data processing pipeline."""
        result = SimulationResult("Data Processing", True)
        
        try:
            from torch.utils.data import DataLoader
            from data.math_dataset_loader import create_demo_dataset, MathDatasetLoader
            from transformers import AutoTokenizer
            
            # Create test data
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            loader = MathDatasetLoader(tokenizer=tokenizer, max_length=self.sim_max_length)
            problems = create_demo_dataset()
            dataset = loader.create_pytorch_dataset(problems, is_training=True)
            
            # Test DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=self.sim_batch_size,
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues in simulation
            )
            
            # Test batch processing
            batch = next(iter(dataloader))
            
            expected_batch_size = min(self.sim_batch_size, len(dataset))
            actual_batch_size = batch['input_ids'].shape[0]
            
            if actual_batch_size != expected_batch_size:
                result.warning_messages.append(
                    f"Batch size mismatch: expected {expected_batch_size}, got {actual_batch_size}"
                )
            
            # Check tensor shapes
            input_ids_shape = batch['input_ids'].shape
            attention_mask_shape = batch['attention_mask'].shape
            
            if input_ids_shape != attention_mask_shape:
                result.warning_messages.append("Input IDs and attention mask shape mismatch")
            
            logger.info(f"Batch processing successful - Shape: {input_ids_shape}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.recommendations.append("Check data processing and batching logic")
        
        return result
    
    def _test_training_setup(self) -> SimulationResult:
        """Test training configuration and setup."""
        result = SimulationResult("Training Setup", True)
        
        try:
            from transformers import TrainingArguments, Trainer
            
            # Test training arguments
            training_args = TrainingArguments(
                output_dir="./test_output",
                num_train_epochs=1,
                max_steps=self.sim_max_steps,
                per_device_train_batch_size=self.sim_batch_size,
                per_device_eval_batch_size=self.sim_batch_size,
                learning_rate=2e-5,
                warmup_steps=2,
                logging_steps=1,
                evaluation_strategy="steps",
                eval_steps=5,
                save_strategy="no",  # Don't save during simulation
                report_to=[],  # Disable wandb for simulation
                remove_unused_columns=False
            )
            
            logger.info("Training arguments created successfully")
            
            # Validate training arguments
            if training_args.max_steps <= 0:
                result.warning_messages.append("Max steps should be positive")
            
            if training_args.per_device_train_batch_size <= 0:
                result.warning_messages.append("Batch size should be positive")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.recommendations.append("Check training configuration parameters")
        
        return result
    
    def _test_training_loop(self) -> SimulationResult:
        """Test training loop execution."""
        result = SimulationResult("Training Loop", True)
        
        if self.quick_mode:
            logger.info("Skipping training loop test in quick mode")
            result.warning_messages.append("Training loop test skipped in quick mode")
            return result
        
        try:
            # Import required modules
            from models.mathematical_reasoning_model import create_mathematical_reasoning_model
            from data.math_dataset_loader import create_demo_dataset, MathDatasetLoader
            from transformers import TrainingArguments, Trainer, AutoTokenizer
            
            # Create small test setup
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            # Create minimal model for testing
            model = create_mathematical_reasoning_model(
                pe_method=self.pe_method,
                base_model="microsoft/DialoGPT-small",
                use_lora=True,
                load_in_4bit=False,
                device_map=None
            )
            
            # Create minimal dataset
            loader = MathDatasetLoader(tokenizer=tokenizer, max_length=256)
            problems = create_demo_dataset()
            train_dataset = loader.create_pytorch_dataset(problems, is_training=True)
            eval_dataset = loader.create_pytorch_dataset(problems, is_training=False)
            
            # Minimal training arguments
            training_args = TrainingArguments(
                output_dir="./test_output",
                max_steps=3,  # Minimal steps
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                learning_rate=1e-4,
                logging_steps=1,
                evaluation_strategy="steps",
                eval_steps=2,
                save_strategy="no",
                report_to=[],
                remove_unused_columns=False,
                dataloader_num_workers=0
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer
            )
            
            # Run minimal training
            logger.info("Starting minimal training test...")
            train_result = trainer.train()
            
            logger.info(f"Training test completed - Final loss: {train_result.training_loss}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            if "out of memory" in str(e).lower():
                result.recommendations.append("Reduce batch size or model size for your hardware")
            else:
                result.recommendations.append("Check training pipeline for configuration issues")
        
        return result
    
    def _test_model_saving(self) -> SimulationResult:
        """Test model saving and loading."""
        result = SimulationResult("Model Saving", True)
        
        try:
            from models.mathematical_reasoning_model import create_mathematical_reasoning_model
            import tempfile
            import shutil
            
            # Create a simple model for testing
            model = create_mathematical_reasoning_model(
                pe_method=self.pe_method,
                base_model="microsoft/DialoGPT-small",
                use_lora=True,
                load_in_4bit=False,
                device_map=None
            )
            
            # Test saving
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = Path(temp_dir) / "test_model"
                model.save_pretrained(str(save_path))
                
                # Check if files were created
                expected_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
                missing_files = []
                
                for file_name in expected_files:
                    if not (save_path / file_name).exists():
                        missing_files.append(file_name)
                
                if missing_files:
                    result.warning_messages.append(f"Missing saved files: {missing_files}")
                
                logger.info(f"Model saved successfully to {save_path}")
                
                # Test loading (optional)
                try:
                    loaded_model = model.__class__.from_pretrained(str(save_path))
                    logger.info("Model loaded successfully")
                except Exception as load_e:
                    result.warning_messages.append(f"Model loading failed: {load_e}")
        
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.recommendations.append("Check model saving implementation")
        
        return result
    
    def _test_evaluation(self) -> SimulationResult:
        """Test evaluation pipeline."""
        result = SimulationResult("Evaluation", True)
        
        try:
            from data.math_dataset_loader import create_demo_dataset
            
            # Test basic evaluation metrics
            problems = create_demo_dataset()
            
            if len(problems) == 0:
                result.warning_messages.append("No problems available for evaluation")
                return result
            
            # Test answer extraction
            for problem in problems:
                if not problem.answer:
                    result.warning_messages.append(f"Empty answer for problem: {problem.problem[:50]}...")
            
            logger.info(f"Evaluation test completed with {len(problems)} problems")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.recommendations.append("Check evaluation pipeline implementation")
        
        return result
    
    def _test_memory_usage(self) -> SimulationResult:
        """Test memory usage patterns."""
        result = SimulationResult("Memory Usage", True)
        
        try:
            import gc
            
            # Get baseline memory
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Simulate memory-intensive operations
            if torch.cuda.is_available():
                # Test GPU memory
                torch.cuda.empty_cache()
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2
                
                # Create some tensors
                test_tensors = []
                for i in range(10):
                    tensor = torch.randn(100, 100).cuda()
                    test_tensors.append(tensor)
                
                peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
                gpu_usage = peak_gpu_memory - initial_gpu_memory
                
                # Cleanup
                del test_tensors
                torch.cuda.empty_cache()
                
                logger.info(f"GPU memory test: {gpu_usage:.1f}MB used")
                
                if gpu_usage > 1000:  # 1GB
                    result.warning_messages.append(f"High GPU memory usage: {gpu_usage:.1f}MB")
            
            # Test CPU memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = current_memory - baseline_memory
            
            logger.info(f"Memory usage test completed - Increase: {memory_increase:.1f}MB")
            
            if memory_increase > 500:  # 500MB
                result.warning_messages.append(f"High memory usage increase: {memory_increase:.1f}MB")
                result.recommendations.append("Monitor memory usage during training")
        
        except Exception as e:
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def _test_hpc_compatibility(self) -> SimulationResult:
        """Test HPC environment compatibility."""
        result = SimulationResult("HPC Compatibility", True)
        
        # Check environment variables
        required_env_vars = ["HOME", "USER"]
        missing_env_vars = [var for var in required_env_vars if var not in os.environ]
        
        if missing_env_vars:
            result.warning_messages.append(f"Missing environment variables: {missing_env_vars}")
        
        # Check common HPC paths
        common_paths = ["/scratch", "/home", "/usr/bin"]
        accessible_paths = [path for path in common_paths if Path(path).exists()]
        
        if "/scratch" not in accessible_paths:
            result.warning_messages.append("No /scratch directory found - may not be on HPC system")
        
        # Check for common HPC commands
        hpc_commands = ["qsub", "sbatch", "module"]
        available_commands = []
        
        for cmd in hpc_commands:
            if shutil.which(cmd):
                available_commands.append(cmd)
        
        if not available_commands:
            result.warning_messages.append("No HPC scheduling commands found")
            result.recommendations.append("This appears to be a local environment, not HPC")
        else:
            logger.info(f"Found HPC commands: {available_commands}")
        
        # Check Python environment
        if "VIRTUAL_ENV" not in os.environ and "CONDA_DEFAULT_ENV" not in os.environ:
            result.warning_messages.append("No virtual environment detected")
            result.recommendations.append("Use virtual environment for package isolation")
        
        return result
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """Compare two version strings. Returns -1, 0, or 1."""
        def version_to_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        v1_tuple = version_to_tuple(version1)
        v2_tuple = version_to_tuple(version2)
        
        if v1_tuple < v2_tuple:
            return -1
        elif v1_tuple > v2_tuple:
            return 1
        else:
            return 0
    
    def _generate_simulation_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation report."""
        
        total_time = time.time() - self.start_time
        
        # Categorize results
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        tests_with_warnings = [r for r in self.results if r.warning_messages]
        
        # Calculate statistics
        success_rate = len(successful_tests) / len(self.results) * 100
        total_warnings = sum(len(r.warning_messages) for r in self.results)
        total_recommendations = sum(len(r.recommendations) for r in self.results)
        
        # Generate report
        report = {
            "simulation_summary": {
                "pe_method": self.pe_method,
                "quick_mode": self.quick_mode,
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": round(success_rate, 1),
                "total_warnings": total_warnings,
                "total_recommendations": total_recommendations,
                "total_execution_time": round(total_time, 2)
            },
            "test_results": [],
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "readiness_assessment": self._assess_readiness()
        }
        
        # Add detailed results
        for result in self.results:
            test_detail = {
                "component": result.component,
                "success": result.success,
                "execution_time": round(result.execution_time, 3),
                "memory_usage": round(result.memory_usage, 2)
            }
            
            if result.error_message:
                test_detail["error"] = result.error_message
                report["critical_issues"].append(f"{result.component}: {result.error_message}")
            
            if result.warning_messages:
                test_detail["warnings"] = result.warning_messages
                report["warnings"].extend([f"{result.component}: {w}" for w in result.warning_messages])
            
            if result.recommendations:
                test_detail["recommendations"] = result.recommendations
                report["recommendations"].extend([f"{result.component}: {r}" for r in result.recommendations])
            
            report["test_results"].append(test_detail)
        
        return report
    
    def _assess_readiness(self) -> Dict[str, Any]:
        """Assess overall readiness for HPC deployment."""
        
        critical_failures = [r for r in self.results if not r.success]
        high_priority_warnings = []
        
        # Identify high-priority warnings
        for result in self.results:
            for warning in result.warning_messages:
                if any(keyword in warning.lower() for keyword in ["memory", "gpu", "cuda", "missing"]):
                    high_priority_warnings.append(f"{result.component}: {warning}")
        
        # Determine readiness level
        if len(critical_failures) == 0:
            if len(high_priority_warnings) == 0:
                readiness_level = "READY"
                readiness_message = "✅ System is ready for HPC deployment"
            else:
                readiness_level = "READY_WITH_WARNINGS"
                readiness_message = "⚠️ System is ready but has warnings that should be addressed"
        else:
            readiness_level = "NOT_READY"
            readiness_message = "❌ System has critical issues that must be fixed before deployment"
        
        return {
            "level": readiness_level,
            "message": readiness_message,
            "critical_failures": len(critical_failures),
            "high_priority_warnings": len(high_priority_warnings),
            "estimated_success_probability": max(0, 100 - len(critical_failures) * 30 - len(high_priority_warnings) * 10)
        }


def print_simulation_report(report: Dict[str, Any]):
    """Print formatted simulation report."""
    
    print("\n" + "="*80)
    print("🔬 EXPERIMENT SIMULATION REPORT")
    print("="*80)
    
    summary = report["simulation_summary"]
    print(f"\n📊 SUMMARY")
    print(f"   PE Method: {summary['pe_method']}")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Success Rate: {summary['success_rate']}%")
    print(f"   Execution Time: {summary['total_execution_time']}s")
    print(f"   Warnings: {summary['total_warnings']}")
    print(f"   Recommendations: {summary['total_recommendations']}")
    
    # Readiness assessment
    readiness = report["readiness_assessment"]
    print(f"\n🎯 READINESS ASSESSMENT")
    print(f"   {readiness['message']}")
    print(f"   Success Probability: {readiness['estimated_success_probability']}%")
    
    # Critical issues
    if report["critical_issues"]:
        print(f"\n❌ CRITICAL ISSUES ({len(report['critical_issues'])})")
        for issue in report["critical_issues"]:
            print(f"   • {issue}")
    
    # Warnings
    if report["warnings"]:
        print(f"\n⚠️ WARNINGS ({len(report['warnings'])})")
        for warning in report["warnings"][:10]:  # Limit to first 10
            print(f"   • {warning}")
        if len(report["warnings"]) > 10:
            print(f"   ... and {len(report['warnings']) - 10} more warnings")
    
    # Recommendations
    if report["recommendations"]:
        print(f"\n💡 RECOMMENDATIONS ({len(report['recommendations'])})")
        for rec in report["recommendations"][:10]:  # Limit to first 10
            print(f"   • {rec}")
        if len(report["recommendations"]) > 10:
            print(f"   ... and {len(report['recommendations']) - 10} more recommendations")
    
    # Test details
    print(f"\n🧪 DETAILED TEST RESULTS")
    for test in report["test_results"]:
        status = "✅" if test["success"] else "❌"
        print(f"   {status} {test['component']:<25} ({test['execution_time']:.2f}s)")
    
    print("\n" + "="*80)


def main():
    """Main simulation function."""
    
    parser = argparse.ArgumentParser(description="Simulate mathematical reasoning experiment")
    parser.add_argument("--pe_method", default="rope", 
                       choices=["rope", "alibi", "sinusoidal", "diet", "t5_relative", "math_adaptive"],
                       help="Positional encoding method to test")
    parser.add_argument("--quick", action="store_true", help="Run quick simulation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--save_report", type=str, help="Save report to JSON file")
    
    args = parser.parse_args()
    
    # Run simulation
    simulator = ExperimentSimulator(
        pe_method=args.pe_method,
        quick_mode=args.quick,
        verbose=args.verbose
    )
    
    report = simulator.run_full_simulation()
    
    # Print report
    print_simulation_report(report)
    
    # Save report if requested
    if args.save_report:
        with open(args.save_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📁 Report saved to: {args.save_report}")
    
    # Exit with appropriate code
    readiness_level = report["readiness_assessment"]["level"]
    if readiness_level == "NOT_READY":
        sys.exit(1)
    elif readiness_level == "READY_WITH_WARNINGS":
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()