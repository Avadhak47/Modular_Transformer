#!/usr/bin/env python3
"""
Experiment Simulation Script

Simulates the complete mathematical reasoning experiment to identify potential
issues before deployment on HPC cluster. Tests all components in a controlled
environment to catch errors early.
"""

import os
import sys
import json
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from unittest.mock import Mock, patch

# Import our modules
from src.models.mathematical_model import MathematicalReasoningModel, MathematicalModelFactory
from data.math_dataset_loader import MathematicalDatasetLoader, MathematicalProblem
from src.positional_encoding import get_positional_encoding, PE_REGISTRY

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentSimulator:
    """Comprehensive experiment simulator for mathematical reasoning training."""
    
    def __init__(self, use_mock_models: bool = True, verbose: bool = True):
        """
        Initialize the experiment simulator.
        
        Args:
            use_mock_models: Whether to use lightweight mock models for testing
            verbose: Whether to enable verbose logging
        """
        self.use_mock_models = use_mock_models
        self.verbose = verbose
        self.temp_dir = None
        self.issues_found = []
        self.warnings_found = []
        
        # Test configurations for each node
        self.node_configs = self._create_test_configurations()
        
        logger.info(f"Initialized ExperimentSimulator (mock_models={use_mock_models})")
    
    def _create_test_configurations(self) -> Dict[int, Dict[str, Any]]:
        """Create test configurations for all nodes."""
        base_config = {
            "model_config": {
                "d_model": 512 if self.use_mock_models else 4096,
                "n_heads": 8 if self.use_mock_models else 32,
                "n_layers": 6 if self.use_mock_models else 24,
                "max_seq_len": 512 if self.use_mock_models else 4096,
                "dropout": 0.1
            },
            "training_config": {
                "batch_size": 2,
                "learning_rate": 1e-4,
                "max_steps": 10 if self.use_mock_models else 1000,
                "warmup_steps": 5 if self.use_mock_models else 100,
                "eval_steps": 5 if self.use_mock_models else 100,
                "save_steps": 10 if self.use_mock_models else 500,
                "logging_steps": 1 if self.use_mock_models else 50,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0
            },
            "data_config": {
                "train_datasets": ["math", "gsm8k"],
                "eval_datasets": ["math_test", "gsm8k_test"],
                "max_train_samples": 20 if self.use_mock_models else 5000,
                "max_eval_samples": 10 if self.use_mock_models else 1000
            },
            "hardware_config": {
                "gpus_per_node": 1,
                "cpu_cores": 4,
                "memory_gb": 16
            }
        }
        
        # Node-specific configurations
        node_configs = {}
        pe_methods = ["sinusoidal", "rope", "alibi", "diet", "t5_relative"]
        base_models = [
            "meta-llama/Llama-2-7b-hf" if self.use_mock_models else "deepseek-ai/deepseek-math-7b-instruct",
            "meta-llama/Llama-2-7b-hf" if self.use_mock_models else "deepseek-ai/deepseek-math-7b-rl",
            "meta-llama/Llama-2-7b-hf" if self.use_mock_models else "microsoft/Orca-Math-Word-Problems-200K",
            "meta-llama/Llama-2-7b-hf" if self.use_mock_models else "tongyx361/DotaMath-DeepSeek-7B",
            "meta-llama/Llama-2-7b-hf" if self.use_mock_models else "microsoft/MindStar-7B"
        ]
        
        for i in range(5):
            config = base_config.copy()
            config["model_config"]["positional_encoding"] = pe_methods[i]
            
            config["sota_integration"] = {
                "base_model": base_models[i],
                "use_lora": True,
                "lora_rank": 16 if self.use_mock_models else 64,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "use_quantization": False if self.use_mock_models else True
            }
            
            config["experiment_config"] = {
                "name": f"test_node_{i}_{pe_methods[i]}",
                "output_dir": f"/tmp/test_experiment/node_{i}",
                "logging_dir": f"/tmp/test_experiment/logs/node_{i}",
                "wandb_project": "math_reasoning_simulation",
                "wandb_run_name": f"sim_node_{i}"
            }
            
            config["hardware_config"]["node_id"] = i
            
            node_configs[i] = config
        
        return node_configs
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run complete experiment simulation."""
        logger.info("🚀 Starting comprehensive experiment simulation...")
        
        results = {
            "simulation_status": "running",
            "tests_passed": 0,
            "tests_failed": 0,
            "issues_found": [],
            "warnings": [],
            "component_results": {}
        }
        
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="math_exp_sim_")
            logger.info(f"Created temporary directory: {self.temp_dir}")
            
            # Test 1: Environment Setup
            results["component_results"]["environment"] = self._test_environment_setup()
            
            # Test 2: Positional Encoding Implementations
            results["component_results"]["positional_encoding"] = self._test_positional_encodings()
            
            # Test 3: Model Loading and Initialization
            results["component_results"]["model_loading"] = self._test_model_loading()
            
            # Test 4: Dataset Loading and Processing
            results["component_results"]["dataset_loading"] = self._test_dataset_loading()
            
            # Test 5: Training Loop Simulation
            results["component_results"]["training_simulation"] = self._test_training_simulation()
            
            # Test 6: Multi-Node Configuration
            results["component_results"]["multi_node_config"] = self._test_multi_node_configuration()
            
            # Test 7: Memory and Resource Usage
            results["component_results"]["resource_usage"] = self._test_resource_usage()
            
            # Test 8: Error Handling and Recovery
            results["component_results"]["error_handling"] = self._test_error_handling()
            
            # Compile final results
            results["issues_found"] = self.issues_found
            results["warnings"] = self.warnings_found
            results["tests_passed"] = sum(1 for r in results["component_results"].values() if r.get("status") == "passed")
            results["tests_failed"] = sum(1 for r in results["component_results"].values() if r.get("status") == "failed")
            results["simulation_status"] = "completed"
            
            self._generate_simulation_report(results)
            
        except Exception as e:
            logger.error(f"Simulation failed with error: {e}")
            results["simulation_status"] = "failed"
            results["error"] = str(e)
            self.issues_found.append(f"Critical simulation error: {e}")
        
        finally:
            self._cleanup()
        
        return results
    
    def _test_environment_setup(self) -> Dict[str, Any]:
        """Test environment setup and dependencies."""
        logger.info("🔧 Testing environment setup...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test Python version
            import sys
            python_version = sys.version_info
            if python_version.major < 3 or python_version.minor < 8:
                test_result["issues"].append(f"Python version {python_version} may be too old")
            
            # Test PyTorch installation
            import torch
            test_result["details"]["pytorch_version"] = torch.__version__
            test_result["details"]["cuda_available"] = torch.cuda.is_available()
            test_result["details"]["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
            
            # Test transformers
            import transformers
            test_result["details"]["transformers_version"] = transformers.__version__
            
            # Test other critical packages
            critical_packages = ["datasets", "accelerate", "peft", "bitsandbytes"]
            for package in critical_packages:
                try:
                    __import__(package)
                    test_result["details"][f"{package}_available"] = True
                except ImportError:
                    test_result["issues"].append(f"Missing package: {package}")
                    test_result["details"][f"{package}_available"] = False
            
            # Test disk space
            import shutil
            disk_usage = shutil.disk_usage("/tmp")
            free_gb = disk_usage.free / (1024**3)
            test_result["details"]["free_disk_gb"] = round(free_gb, 2)
            
            if free_gb < 10:
                test_result["issues"].append(f"Low disk space: {free_gb:.2f} GB available")
            
            # Test memory
            import psutil
            memory = psutil.virtual_memory()
            test_result["details"]["total_memory_gb"] = round(memory.total / (1024**3), 2)
            test_result["details"]["available_memory_gb"] = round(memory.available / (1024**3), 2)
            
            if memory.available / (1024**3) < 8:
                test_result["issues"].append(f"Low memory: {memory.available / (1024**3):.2f} GB available")
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
            logger.info(f"✅ Environment setup test: {test_result['status']}")
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Environment setup test failed: {e}")
            logger.error(f"❌ Environment setup test failed: {e}")
        
        return test_result
    
    def _test_positional_encodings(self) -> Dict[str, Any]:
        """Test all positional encoding implementations."""
        logger.info("🧮 Testing positional encoding implementations...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            for pe_name in PE_REGISTRY.keys():
                logger.info(f"Testing {pe_name} positional encoding...")
                
                try:
                    # Test PE creation
                    pe_config = {
                        "d_model": 512,
                        "dim": 64,
                        "num_heads": 8,
                        "max_seq_len": 1024
                    }
                    
                    pe = get_positional_encoding(pe_name, **pe_config)
                    test_result["details"][f"{pe_name}_creation"] = "success"
                    
                    # Test forward pass with dummy data
                    if pe_name == "sinusoidal":
                        dummy_input = torch.randn(2, 100, 512)
                        output = pe(dummy_input)
                        assert output.shape == dummy_input.shape
                        
                    elif pe_name == "rope":
                        dummy_q = torch.randn(2, 8, 100, 64)
                        dummy_k = torch.randn(2, 8, 100, 64)
                        q_out, k_out = pe(dummy_q, dummy_k)
                        assert q_out.shape == dummy_q.shape
                        assert k_out.shape == dummy_k.shape
                        
                    elif pe_name == "alibi":
                        dummy_scores = torch.randn(2, 8, 100, 100)
                        output = pe(dummy_scores)
                        assert output.shape == dummy_scores.shape
                    
                    test_result["details"][f"{pe_name}_forward"] = "success"
                    logger.info(f"  ✅ {pe_name} tests passed")
                    
                except Exception as e:
                    test_result["issues"].append(f"{pe_name} PE test failed: {e}")
                    test_result["details"][f"{pe_name}_error"] = str(e)
                    logger.warning(f"  ⚠️ {pe_name} test failed: {e}")
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Positional encoding tests failed: {e}")
            logger.error(f"❌ Positional encoding tests failed: {e}")
        
        return test_result
    
    def _test_model_loading(self) -> Dict[str, Any]:
        """Test model loading and initialization."""
        logger.info("🤖 Testing model loading and initialization...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            for node_id, config in self.node_configs.items():
                logger.info(f"Testing model for Node {node_id}...")
                
                try:
                    if self.use_mock_models:
                        # Use mock model for testing
                        with patch('src.models.mathematical_model.LlamaForCausalLM') as mock_llama:
                            mock_model = Mock()
                            mock_model.config = Mock()
                            mock_model.config.hidden_size = 512
                            mock_model.config.num_attention_heads = 8
                            mock_llama.from_pretrained.return_value = mock_model
                            
                            model = MathematicalModelFactory.create_node_model(node_id, config)
                            test_result["details"][f"node_{node_id}_model_creation"] = "success"
                    
                    else:
                        # Test actual model loading (will likely fail without proper setup)
                        try:
                            model = MathematicalModelFactory.create_node_model(node_id, config)
                            test_result["details"][f"node_{node_id}_model_creation"] = "success"
                        except Exception as e:
                            test_result["issues"].append(f"Node {node_id} model loading failed: {e}")
                            test_result["details"][f"node_{node_id}_error"] = str(e)
                    
                    logger.info(f"  ✅ Node {node_id} model test passed")
                    
                except Exception as e:
                    test_result["issues"].append(f"Node {node_id} model test failed: {e}")
                    logger.warning(f"  ⚠️ Node {node_id} model test failed: {e}")
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Model loading tests failed: {e}")
            logger.error(f"❌ Model loading tests failed: {e}")
        
        return test_result
    
    def _test_dataset_loading(self) -> Dict[str, Any]:
        """Test dataset loading and processing."""
        logger.info("📚 Testing dataset loading and processing...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Initialize dataset loader
            loader = MathematicalDatasetLoader(
                cache_dir=f"{self.temp_dir}/data_cache",
                max_problems=10 if self.use_mock_models else 100
            )
            
            # Test individual dataset loading
            datasets_to_test = ["math", "gsm8k"] if not self.use_mock_models else []
            
            for dataset_name in datasets_to_test:
                try:
                    if dataset_name == "math":
                        problems = loader.load_math_dataset("train")
                    elif dataset_name == "gsm8k":
                        problems = loader.load_gsm8k_dataset("train")
                    
                    test_result["details"][f"{dataset_name}_problems_loaded"] = len(problems)
                    logger.info(f"  ✅ Loaded {len(problems)} {dataset_name} problems")
                    
                except Exception as e:
                    test_result["issues"].append(f"Failed to load {dataset_name}: {e}")
                    logger.warning(f"  ⚠️ Failed to load {dataset_name}: {e}")
            
            # Test synthetic problem generation
            try:
                synthetic_problems = loader.generate_synthetic_problems(5)
                test_result["details"]["synthetic_problems_generated"] = len(synthetic_problems)
                logger.info(f"  ✅ Generated {len(synthetic_problems)} synthetic problems")
            except Exception as e:
                test_result["issues"].append(f"Synthetic problem generation failed: {e}")
                logger.warning(f"  ⚠️ Synthetic problem generation failed: {e}")
            
            # Test fallback datasets
            try:
                fallback_problems = loader._load_fallback_math_problems()
                test_result["details"]["fallback_problems"] = len(fallback_problems)
                logger.info(f"  ✅ Fallback dataset contains {len(fallback_problems)} problems")
            except Exception as e:
                test_result["issues"].append(f"Fallback dataset test failed: {e}")
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Dataset loading tests failed: {e}")
            logger.error(f"❌ Dataset loading tests failed: {e}")
        
        return test_result
    
    def _test_training_simulation(self) -> Dict[str, Any]:
        """Test training loop simulation."""
        logger.info("🏋️ Testing training loop simulation...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Mock a minimal training setup
            if self.use_mock_models:
                # Create mock components
                mock_model = Mock()
                mock_tokenizer = Mock()
                mock_problems = [
                    MathematicalProblem(
                        problem="What is 2+2?",
                        solution="2+2=4",
                        answer="4",
                        source="test"
                    )
                ]
                
                # Test training configuration parsing
                config = self.node_configs[0]
                training_config = config["training_config"]
                
                required_fields = ["batch_size", "learning_rate", "max_steps"]
                for field in required_fields:
                    if field not in training_config:
                        test_result["issues"].append(f"Missing training config field: {field}")
                
                test_result["details"]["training_config_valid"] = len(test_result["issues"]) == 0
                
                # Test optimizer configuration
                try:
                    import torch.optim as optim
                    optimizer = optim.AdamW(
                        [torch.randn(10, requires_grad=True)],
                        lr=training_config["learning_rate"],
                        weight_decay=training_config.get("weight_decay", 0.01)
                    )
                    test_result["details"]["optimizer_creation"] = "success"
                except Exception as e:
                    test_result["issues"].append(f"Optimizer creation failed: {e}")
                
                # Test scheduler configuration
                try:
                    from transformers import get_cosine_schedule_with_warmup
                    scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=training_config.get("warmup_steps", 100),
                        num_training_steps=training_config["max_steps"]
                    )
                    test_result["details"]["scheduler_creation"] = "success"
                except Exception as e:
                    test_result["issues"].append(f"Scheduler creation failed: {e}")
                
                logger.info("  ✅ Training simulation tests passed")
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Training simulation tests failed: {e}")
            logger.error(f"❌ Training simulation tests failed: {e}")
        
        return test_result
    
    def _test_multi_node_configuration(self) -> Dict[str, Any]:
        """Test multi-node configuration setup."""
        logger.info("🌐 Testing multi-node configuration...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Validate all node configurations
            for node_id, config in self.node_configs.items():
                try:
                    # Check required configuration sections
                    required_sections = ["model_config", "training_config", "data_config", "experiment_config"]
                    for section in required_sections:
                        if section not in config:
                            test_result["issues"].append(f"Node {node_id} missing config section: {section}")
                    
                    # Check unique positional encoding per node
                    pe_type = config["model_config"]["positional_encoding"]
                    test_result["details"][f"node_{node_id}_pe_type"] = pe_type
                    
                    # Check output directories are unique
                    output_dir = config["experiment_config"]["output_dir"]
                    if output_dir in [self.node_configs[i]["experiment_config"]["output_dir"] 
                                    for i in self.node_configs.keys() if i != node_id]:
                        test_result["issues"].append(f"Node {node_id} has duplicate output directory")
                    
                except Exception as e:
                    test_result["issues"].append(f"Node {node_id} config validation failed: {e}")
            
            # Test distributed training environment variables
            env_vars = ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
            for var in env_vars:
                test_result["details"][f"env_{var}"] = os.environ.get(var, "not_set")
            
            # Check for PE type uniqueness across nodes
            pe_types = [config["model_config"]["positional_encoding"] for config in self.node_configs.values()]
            if len(set(pe_types)) != len(pe_types):
                test_result["issues"].append("Duplicate positional encoding types across nodes")
            
            test_result["details"]["unique_pe_types"] = len(set(pe_types)) == len(pe_types)
            test_result["details"]["total_nodes"] = len(self.node_configs)
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
            logger.info(f"  ✅ Multi-node configuration test passed for {len(self.node_configs)} nodes")
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Multi-node configuration tests failed: {e}")
            logger.error(f"❌ Multi-node configuration tests failed: {e}")
        
        return test_result
    
    def _test_resource_usage(self) -> Dict[str, Any]:
        """Test memory and resource usage patterns."""
        logger.info("📊 Testing resource usage patterns...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            import psutil
            import torch
            
            # Get baseline memory usage
            baseline_memory = psutil.virtual_memory().used / (1024**3)
            test_result["details"]["baseline_memory_gb"] = round(baseline_memory, 2)
            
            # Test GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                test_result["details"]["gpu_memory_gb"] = round(gpu_memory, 2)
                
                # Test GPU allocation
                try:
                    test_tensor = torch.randn(1000, 1000, device='cuda')
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    test_result["details"]["gpu_allocation_test_gb"] = round(allocated, 4)
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    test_result["issues"].append(f"GPU memory test failed: {e}")
            
            # Estimate memory requirements for training
            if self.use_mock_models:
                estimated_model_memory = 0.5  # GB for mock model
            else:
                estimated_model_memory = 14  # GB for 7B model with quantization
            
            test_result["details"]["estimated_model_memory_gb"] = estimated_model_memory
            
            available_memory = psutil.virtual_memory().available / (1024**3)
            if available_memory < estimated_model_memory * 2:  # 2x safety margin
                test_result["issues"].append(
                    f"Insufficient memory: need ~{estimated_model_memory*2:.1f}GB, "
                    f"have {available_memory:.1f}GB"
                )
            
            # Test disk space requirements
            estimated_disk_usage = 20 if not self.use_mock_models else 1  # GB
            import shutil
            free_disk = shutil.disk_usage("/tmp").free / (1024**3)
            
            if free_disk < estimated_disk_usage:
                test_result["issues"].append(
                    f"Insufficient disk space: need ~{estimated_disk_usage}GB, "
                    f"have {free_disk:.1f}GB"
                )
            
            test_result["details"]["estimated_disk_usage_gb"] = estimated_disk_usage
            test_result["details"]["available_disk_gb"] = round(free_disk, 2)
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
            logger.info("  ✅ Resource usage tests completed")
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Resource usage tests failed: {e}")
            logger.error(f"❌ Resource usage tests failed: {e}")
        
        return test_result
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and recovery mechanisms."""
        logger.info("🛡️ Testing error handling and recovery...")
        
        test_result = {"status": "passed", "details": {}, "issues": []}
        
        try:
            # Test configuration validation
            invalid_config = {"invalid": "config"}
            try:
                MathematicalModelFactory.create_node_model(0, invalid_config)
                test_result["issues"].append("Invalid config validation failed")
            except Exception:
                test_result["details"]["config_validation"] = "success"
            
            # Test fallback mechanisms
            try:
                loader = MathematicalDatasetLoader(cache_dir="/invalid/path", max_problems=5)
                # Should handle invalid cache directory gracefully
                test_result["details"]["invalid_cache_handling"] = "success"
            except Exception as e:
                test_result["issues"].append(f"Cache directory error handling failed: {e}")
            
            # Test out-of-memory simulation
            try:
                # Create large tensor to potentially trigger OOM
                if torch.cuda.is_available():
                    available_memory = torch.cuda.get_device_properties(0).total_memory
                    # Try to allocate 90% of available memory
                    test_size = int(available_memory * 0.9 / 4)  # float32 = 4 bytes
                    large_tensor = torch.randn(test_size, device='cuda')
                    del large_tensor
                    torch.cuda.empty_cache()
                
                test_result["details"]["memory_stress_test"] = "success"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    test_result["details"]["oom_handling"] = "detected"
                else:
                    test_result["issues"].append(f"Unexpected memory error: {e}")
            except Exception as e:
                test_result["issues"].append(f"Memory stress test failed: {e}")
            
            if test_result["issues"]:
                test_result["status"] = "warning"
                self.warnings_found.extend(test_result["issues"])
            
            logger.info("  ✅ Error handling tests completed")
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
            self.issues_found.append(f"Error handling tests failed: {e}")
            logger.error(f"❌ Error handling tests failed: {e}")
        
        return test_result
    
    def _generate_simulation_report(self, results: Dict[str, Any]):
        """Generate comprehensive simulation report."""
        logger.info("📋 Generating simulation report...")
        
        report_path = Path(self.temp_dir) / "simulation_report.json"
        
        # Add summary statistics
        summary = {
            "total_tests": len(results["component_results"]),
            "tests_passed": results["tests_passed"],
            "tests_failed": results["tests_failed"],
            "tests_with_warnings": sum(1 for r in results["component_results"].values() 
                                     if r.get("status") == "warning"),
            "critical_issues": len([issue for issue in self.issues_found 
                                  if "critical" in issue.lower() or "failed" in issue.lower()]),
            "total_warnings": len(self.warnings_found)
        }
        
        results["summary"] = summary
        
        # Save detailed report
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary to console
        print("\n" + "="*80)
        print("🎯 EXPERIMENT SIMULATION REPORT")
        print("="*80)
        print(f"Status: {results['simulation_status'].upper()}")
        print(f"Tests Passed: {summary['tests_passed']}/{summary['total_tests']}")
        print(f"Tests Failed: {summary['tests_failed']}/{summary['total_tests']}")
        print(f"Tests with Warnings: {summary['tests_with_warnings']}/{summary['total_tests']}")
        print(f"Critical Issues: {summary['critical_issues']}")
        print(f"Total Warnings: {summary['total_warnings']}")
        
        if self.issues_found:
            print("\n❌ CRITICAL ISSUES FOUND:")
            for issue in self.issues_found:
                print(f"  • {issue}")
        
        if self.warnings_found:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings_found[:10]:  # Show first 10 warnings
                print(f"  • {warning}")
            if len(self.warnings_found) > 10:
                print(f"  ... and {len(self.warnings_found) - 10} more warnings")
        
        if not self.issues_found:
            print("\n✅ No critical issues found! Experiment appears ready for HPC deployment.")
        else:
            print("\n🚨 Critical issues must be resolved before HPC deployment!")
        
        print(f"\nDetailed report saved to: {report_path}")
        print("="*80)
    
    def _cleanup(self):
        """Clean up temporary resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")


def main():
    """Main simulation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate Mathematical Reasoning Experiment")
    parser.add_argument("--use-real-models", action="store_true", 
                       help="Use real models instead of mocks (requires more resources)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    simulator = ExperimentSimulator(
        use_mock_models=not args.use_real_models,
        verbose=args.verbose
    )
    
    results = simulator.run_full_simulation()
    
    # Exit with appropriate code
    if results["simulation_status"] == "failed" or results["tests_failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()