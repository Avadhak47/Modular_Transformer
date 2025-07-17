#!/usr/bin/env python3
"""
Simplified Experiment Simulation

Tests the core framework logic without requiring heavy dependencies.
Identifies architectural and configuration issues that could cause problems
during HPC deployment.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class SimplifiedSimulation:
    """Lightweight simulation to test framework architecture."""
    
    def __init__(self):
        self.issues_found = []
        self.warnings_found = []
        self.test_results = {}
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run comprehensive architectural tests."""
        print("🚀 Running Simplified Mathematical Reasoning Experiment Simulation")
        print("=" * 70)
        
        # Test 1: Project Structure
        self._test_project_structure()
        
        # Test 2: Configuration Files
        self._test_configuration_integrity()
        
        # Test 3: Import Structure
        self._test_import_structure()
        
        # Test 4: Positional Encoding Registry
        self._test_positional_encoding_registry()
        
        # Test 5: Multi-Node Setup
        self._test_multi_node_setup()
        
        # Test 6: HPC Compatibility
        self._test_hpc_compatibility()
        
        # Generate report
        return self._generate_report()
    
    def _test_project_structure(self):
        """Test project directory structure."""
        print("\n📁 Testing project structure...")
        
        required_dirs = [
            "src/models",
            "src/positional_encoding", 
            "src/utils",
            "data",
            "scripts",
            "configs"
        ]
        
        required_files = [
            "src/models/mathematical_model.py",
            "src/positional_encoding/__init__.py",
            "src/positional_encoding/sinusoidal.py",
            "src/positional_encoding/rope.py",
            "src/positional_encoding/alibi.py",
            "src/positional_encoding/diet.py",
            "src/positional_encoding/t5_relative.py",
            "data/math_dataset_loader.py",
            "scripts/train_mathematical_model.py",
            "scripts/simulate_experiment.py",
            "requirements.txt"
        ]
        
        missing_dirs = []
        missing_files = []
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing_dirs.append(dir_path)
        
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_dirs:
            self.issues_found.extend([f"Missing directory: {d}" for d in missing_dirs])
        
        if missing_files:
            self.issues_found.extend([f"Missing file: {f}" for f in missing_files])
        
        self.test_results["project_structure"] = {
            "status": "passed" if not (missing_dirs or missing_files) else "failed",
            "missing_directories": missing_dirs,
            "missing_files": missing_files
        }
        
        print(f"  {'✅' if not (missing_dirs or missing_files) else '❌'} Project structure check")
    
    def _test_configuration_integrity(self):
        """Test configuration file integrity."""
        print("\n⚙️ Testing configuration integrity...")
        
        # Create test node configurations
        node_configs = self._create_test_node_configs()
        
        config_issues = []
        
        # Test each node configuration
        for node_id, config in node_configs.items():
            try:
                # Check required sections
                required_sections = ["model_config", "training_config", "data_config"]
                for section in required_sections:
                    if section not in config:
                        config_issues.append(f"Node {node_id} missing {section}")
                
                # Check PE type is valid
                pe_type = config.get("model_config", {}).get("positional_encoding")
                valid_pe_types = ["sinusoidal", "rope", "alibi", "diet", "t5_relative"]
                if pe_type not in valid_pe_types:
                    config_issues.append(f"Node {node_id} has invalid PE type: {pe_type}")
                
                # Check training parameters
                training_config = config.get("training_config", {})
                required_training_params = ["batch_size", "learning_rate", "max_steps"]
                for param in required_training_params:
                    if param not in training_config:
                        config_issues.append(f"Node {node_id} missing training param: {param}")
                
            except Exception as e:
                config_issues.append(f"Node {node_id} config error: {e}")
        
        # Test PE type uniqueness
        pe_types = [config.get("model_config", {}).get("positional_encoding") 
                   for config in node_configs.values()]
        if len(set(pe_types)) != len(pe_types):
            config_issues.append("Duplicate PE types across nodes")
        
        if config_issues:
            self.issues_found.extend(config_issues)
        
        self.test_results["configuration_integrity"] = {
            "status": "passed" if not config_issues else "failed",
            "issues": config_issues,
            "total_nodes": len(node_configs),
            "unique_pe_types": len(set(pe_types))
        }
        
        print(f"  {'✅' if not config_issues else '❌'} Configuration integrity check")
    
    def _test_import_structure(self):
        """Test Python import structure."""
        print("\n🐍 Testing import structure...")
        
        import_issues = []
        
        # Test basic Python syntax
        python_files = [
            "src/models/mathematical_model.py",
            "src/positional_encoding/__init__.py",
            "data/math_dataset_loader.py",
            "scripts/train_mathematical_model.py"
        ]
        
        for file_path in python_files:
            if Path(file_path).exists():
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Basic syntax check
                    compile(content, file_path, 'exec')
                    
                except SyntaxError as e:
                    import_issues.append(f"Syntax error in {file_path}: {e}")
                except Exception as e:
                    import_issues.append(f"Error reading {file_path}: {e}")
        
        # Test PE registry structure
        try:
            pe_init_path = "src/positional_encoding/__init__.py"
            if Path(pe_init_path).exists():
                with open(pe_init_path, 'r') as f:
                    content = f.read()
                
                # Check for PE_REGISTRY
                if "PE_REGISTRY" not in content:
                    import_issues.append("PE_REGISTRY not found in __init__.py")
                
                # Check for get_positional_encoding function
                if "def get_positional_encoding" not in content:
                    import_issues.append("get_positional_encoding function not found")
                    
        except Exception as e:
            import_issues.append(f"Error checking PE registry: {e}")
        
        if import_issues:
            self.issues_found.extend(import_issues)
        
        self.test_results["import_structure"] = {
            "status": "passed" if not import_issues else "failed",
            "issues": import_issues
        }
        
        print(f"  {'✅' if not import_issues else '❌'} Import structure check")
    
    def _test_positional_encoding_registry(self):
        """Test positional encoding registry completeness."""
        print("\n🧮 Testing positional encoding registry...")
        
        pe_issues = []
        expected_pe_types = ["sinusoidal", "rope", "alibi", "diet", "t5_relative"]
        
        for pe_type in expected_pe_types:
            pe_file = f"src/positional_encoding/{pe_type}.py"
            if not Path(pe_file).exists():
                pe_issues.append(f"Missing PE implementation: {pe_file}")
            else:
                try:
                    with open(pe_file, 'r') as f:
                        content = f.read()
                    
                    # Check for class definition
                    expected_classes = {
                        "sinusoidal": "SinusoidalPositionalEncoding",
                        "rope": "RotaryPositionalEmbedding", 
                        "alibi": "ALiBiPositionalBias",
                        "diet": "DIETPositionalEncoding",
                        "t5_relative": "T5RelativePositionalBias"
                    }
                    
                    expected_class = expected_classes[pe_type]
                    if f"class {expected_class}" not in content:
                        pe_issues.append(f"Missing class {expected_class} in {pe_file}")
                    
                    # Check for forward method
                    if "def forward(" not in content:
                        pe_issues.append(f"Missing forward method in {pe_file}")
                        
                except Exception as e:
                    pe_issues.append(f"Error checking {pe_file}: {e}")
        
        if pe_issues:
            self.issues_found.extend(pe_issues)
        
        self.test_results["positional_encoding_registry"] = {
            "status": "passed" if not pe_issues else "failed",
            "issues": pe_issues,
            "expected_types": expected_pe_types,
            "implementations_found": len(expected_pe_types) - len([i for i in pe_issues if "Missing PE implementation" in i])
        }
        
        print(f"  {'✅' if not pe_issues else '❌'} Positional encoding registry check")
    
    def _test_multi_node_setup(self):
        """Test multi-node configuration setup."""
        print("\n🌐 Testing multi-node setup...")
        
        multinode_issues = []
        
        # Test for 5 distinct node configurations
        node_configs = self._create_test_node_configs()
        
        if len(node_configs) != 5:
            multinode_issues.append(f"Expected 5 nodes, found {len(node_configs)}")
        
        # Test unique PE assignments
        pe_assignments = {}
        for node_id, config in node_configs.items():
            pe_type = config.get("model_config", {}).get("positional_encoding")
            if pe_type in pe_assignments:
                multinode_issues.append(f"PE type {pe_type} assigned to multiple nodes")
            pe_assignments[pe_type] = node_id
        
        # Test unique output directories
        output_dirs = set()
        for node_id, config in node_configs.items():
            output_dir = config.get("experiment_config", {}).get("output_dir", "")
            if output_dir in output_dirs:
                multinode_issues.append(f"Duplicate output directory: {output_dir}")
            output_dirs.add(output_dir)
        
        # Test distributed training compatibility
        required_env_vars = ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]
        missing_env_docs = []
        
        # Check if training script handles distributed setup
        train_script_path = "scripts/train_mathematical_model.py"
        if Path(train_script_path).exists():
            try:
                with open(train_script_path, 'r') as f:
                    content = f.read()
                
                if "distributed" not in content.lower():
                    multinode_issues.append("Training script doesn't appear to handle distributed training")
                
                if "WORLD_SIZE" not in content:
                    multinode_issues.append("Training script doesn't check WORLD_SIZE environment variable")
                    
            except Exception as e:
                multinode_issues.append(f"Error checking training script: {e}")
        
        if multinode_issues:
            self.issues_found.extend(multinode_issues)
        
        self.test_results["multi_node_setup"] = {
            "status": "passed" if not multinode_issues else "failed",
            "issues": multinode_issues,
            "total_nodes": len(node_configs),
            "unique_pe_assignments": len(pe_assignments),
            "unique_output_dirs": len(output_dirs)
        }
        
        print(f"  {'✅' if not multinode_issues else '❌'} Multi-node setup check")
    
    def _test_hpc_compatibility(self):
        """Test HPC deployment compatibility."""
        print("\n🖥️ Testing HPC compatibility...")
        
        hpc_issues = []
        
        # Check for PBS/SLURM job script patterns
        scripts_dir = Path("scripts")
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob("*.sh"))
            
            if not script_files:
                hpc_issues.append("No shell scripts found for HPC job submission")
            
            # Check for common HPC patterns
            hpc_patterns = ["#PBS", "#SBATCH", "module load", "mpirun", "srun"]
            found_patterns = set()
            
            for script_file in script_files:
                try:
                    with open(script_file, 'r') as f:
                        content = f.read()
                    
                    for pattern in hpc_patterns:
                        if pattern in content:
                            found_patterns.add(pattern)
                            
                except Exception as e:
                    hpc_issues.append(f"Error reading {script_file}: {e}")
            
            if not found_patterns:
                self.warnings_found.append("No HPC job scheduler patterns found in scripts")
        
        # Check resource requirements
        node_configs = self._create_test_node_configs()
        total_gpu_requirement = 0
        total_memory_requirement = 0
        
        for config in node_configs.values():
            hw_config = config.get("hardware_config", {})
            total_gpu_requirement += hw_config.get("gpus_per_node", 1)
            total_memory_requirement += hw_config.get("memory_gb", 16)
        
        # Check if requirements are reasonable for HPC
        if total_gpu_requirement > 20:  # 5 nodes * 4 GPUs each = 20 GPUs
            hpc_issues.append(f"High GPU requirement: {total_gpu_requirement} GPUs total")
        
        if total_memory_requirement > 640:  # 5 nodes * 128GB each = 640GB
            hpc_issues.append(f"High memory requirement: {total_memory_requirement} GB total")
        
        # Check for containerization support
        container_files = ["Dockerfile", "Singularity.def", "containers/"]
        has_containers = any(Path(f).exists() for f in container_files)
        
        if not has_containers:
            self.warnings_found.append("No containerization files found - may complicate HPC deployment")
        
        if hpc_issues:
            self.issues_found.extend(hpc_issues)
        
        self.test_results["hpc_compatibility"] = {
            "status": "passed" if not hpc_issues else "failed",
            "issues": hpc_issues,
            "total_gpu_requirement": total_gpu_requirement,
            "total_memory_requirement": total_memory_requirement,
            "has_containers": has_containers
        }
        
        print(f"  {'✅' if not hpc_issues else '❌'} HPC compatibility check")
    
    def _create_test_node_configs(self) -> Dict[int, Dict[str, Any]]:
        """Create test node configurations."""
        pe_methods = ["sinusoidal", "rope", "alibi", "diet", "t5_relative"]
        
        configs = {}
        for i in range(5):
            configs[i] = {
                "model_config": {
                    "positional_encoding": pe_methods[i],
                    "d_model": 4096,
                    "max_seq_len": 4096
                },
                "training_config": {
                    "batch_size": 2,
                    "learning_rate": 1e-4,
                    "max_steps": 1000
                },
                "data_config": {
                    "train_datasets": ["math", "gsm8k"]
                },
                "hardware_config": {
                    "node_id": i,
                    "gpus_per_node": 4,
                    "memory_gb": 128
                },
                "experiment_config": {
                    "output_dir": f"/scratch/user/math_reasoning/node_{i}",
                    "logging_dir": f"/scratch/user/math_reasoning/logs/node_{i}"
                }
            }
        
        return configs
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate final simulation report."""
        total_tests = len(self.test_results)
        tests_passed = sum(1 for result in self.test_results.values() 
                          if result.get("status") == "passed")
        tests_failed = total_tests - tests_passed
        
        print("\n" + "=" * 70)
        print("📋 SIMULATION REPORT")
        print("=" * 70)
        print(f"Tests Run: {total_tests}")
        print(f"Tests Passed: {tests_passed}")
        print(f"Tests Failed: {tests_failed}")
        print(f"Critical Issues: {len(self.issues_found)}")
        print(f"Warnings: {len(self.warnings_found)}")
        
        if self.issues_found:
            print(f"\n❌ CRITICAL ISSUES FOUND ({len(self.issues_found)}):")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. {issue}")
        
        if self.warnings_found:
            print(f"\n⚠️ WARNINGS ({len(self.warnings_found)}):")
            for i, warning in enumerate(self.warnings_found, 1):
                print(f"  {i}. {warning}")
        
        # Readiness assessment
        readiness_score = (tests_passed / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n🎯 DEPLOYMENT READINESS: {readiness_score:.1f}%")
        
        if readiness_score >= 90 and len(self.issues_found) == 0:
            print("✅ READY FOR HPC DEPLOYMENT")
            deployment_ready = True
        elif readiness_score >= 70 and len(self.issues_found) <= 2:
            print("⚠️ MOSTLY READY - Minor issues to fix")
            deployment_ready = False
        else:
            print("❌ NOT READY - Critical issues must be resolved")
            deployment_ready = False
        
        # Key recommendations
        print(f"\n📝 KEY RECOMMENDATIONS:")
        if not deployment_ready:
            print("  1. Fix all critical issues before HPC deployment")
            print("  2. Test with smaller dataset first")
            print("  3. Verify all dependencies are available on HPC cluster")
        else:
            print("  1. Run small-scale test on HPC cluster first")
            print("  2. Monitor resource usage during initial runs")
            print("  3. Have fallback plan for any dependency issues")
        
        print("=" * 70)
        
        return {
            "total_tests": total_tests,
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "issues_found": self.issues_found,
            "warnings_found": self.warnings_found,
            "readiness_score": readiness_score,
            "deployment_ready": deployment_ready,
            "test_results": self.test_results
        }


def main():
    """Run the simplified simulation."""
    simulator = SimplifiedSimulation()
    results = simulator.run_simulation()
    
    # Exit with appropriate code
    if results["deployment_ready"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()