#!/usr/bin/env python3
"""
Enhanced SOTA Mathematical Dataset Loader for Multi-Node HPC Training
Integrates: DeepSeekMath, InternLM-Math, Orca-Math, DotaMath datasets and techniques
Author: Avadhesh Kumar (2024EET2799)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import re
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from transformers import AutoTokenizer
import sympy
from sympy import latex, simplify, parse_expr

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MathematicalProblem:
    """Enhanced mathematical problem with SOTA features."""
    problem: str
    solution: str
    final_answer: str
    reasoning_steps: List[str] = None
    difficulty_level: str = "medium"
    problem_type: str = "general"
    source_dataset: str = "unknown"
    
    # SOTA-specific fields
    chain_of_thought: str = ""
    code_solution: str = ""
    verification_code: str = ""
    decomposition_steps: List[str] = None
    multi_agent_variants: List[str] = None
    formal_proof: str = ""
    
    # Meta information
    token_count: int = 0
    estimated_difficulty: float = 0.5
    mathematical_concepts: List[str] = None
    requires_code: bool = False

class SOTAMathDatasetLoader:
    """Enhanced dataset loader integrating multiple SOTA mathematical reasoning approaches."""
    
    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        sota_method: str = "deepseekmath",
        use_chain_of_thought: bool = True,
        use_code_assistance: bool = False,
        use_multi_agent_learning: bool = False,
        max_length: int = 4096,
        cache_dir: Optional[str] = None
    ):
        """Initialize the SOTA dataset loader."""
        self.tokenizer = tokenizer
        self.sota_method = sota_method
        self.use_chain_of_thought = use_chain_of_thought
        self.use_code_assistance = use_code_assistance
        self.use_multi_agent_learning = use_multi_agent_learning
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # SOTA-specific configurations
        self.sota_configs = {
            "deepseekmath": {
                "use_continued_pretraining": True,
                "grpo_data_ratio": 0.7,
                "math_corpus_size": 120_000_000_000,  # 120B tokens
                "problem_augmentation": True
            },
            "orca_math": {
                "multi_agent_generation": True,
                "iterative_preference_learning": True,
                "synthetic_data_ratio": 0.6,
                "agent_count": 3
            },
            "dotamath": {
                "decomposition_reasoning": True,
                "code_assistance": True,
                "self_correction": True,
                "step_level_verification": True
            },
            "internlm_math": {
                "formal_reasoning": True,
                "lean_integration": True,
                "verifiable_proofs": True,
                "symbolic_computation": True
            },
            "mindstar": {
                "inference_optimization": True,
                "real_time_reasoning": True,
                "attention_analysis": True,
                "efficiency_focus": True
            }
        }
        
        self.current_config = self.sota_configs.get(sota_method, {})
        logger.info(f"Initialized SOTA dataset loader with method: {sota_method}")
        
    def load_base_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load base mathematical reasoning datasets."""
        logger.info("Loading base mathematical reasoning datasets...")
        
        datasets = {}
        
        # Load MATH dataset
        try:
            datasets["math"] = self._load_math_dataset()
            logger.info(f"Loaded MATH dataset: {len(datasets['math'])} problems")
        except Exception as e:
            logger.warning(f"Failed to load MATH dataset: {e}")
            datasets["math"] = []
        
        # Load GSM8K dataset
        try:
            datasets["gsm8k"] = self._load_gsm8k_dataset()
            logger.info(f"Loaded GSM8K dataset: {len(datasets['gsm8k'])} problems")
        except Exception as e:
            logger.warning(f"Failed to load GSM8K dataset: {e}")
            datasets["gsm8k"] = []
        
        return datasets
    
    def _load_math_dataset(self, split: str = "train") -> List[MathematicalProblem]:
        """Load and process MATH dataset."""
        cache_file = self.cache_dir / f"math_{split}_processed.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached MATH dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            # Load from HuggingFace
            dataset = load_dataset("hendrycks/competition_math", split=split)
            
            problems = []
            for item in dataset:
                problem = MathematicalProblem(
                    problem=item["problem"],
                    solution=item["solution"],
                    final_answer=self._extract_final_answer(item["solution"]),
                    difficulty_level=item.get("level", "unknown"),
                    problem_type=item.get("type", "general"),
                    source_dataset="math",
                    mathematical_concepts=self._extract_mathematical_concepts(item["problem"]),
                    token_count=len(item["problem"].split()) + len(item["solution"].split())
                )
                
                # Enhance with SOTA techniques
                problem = self._enhance_with_sota_techniques(problem)
                problems.append(problem)
            
            # Cache processed dataset
            with open(cache_file, 'wb') as f:
                pickle.dump(problems, f)
            
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return self._load_local_math_dataset(split)
    
    def _load_gsm8k_dataset(self, split: str = "train") -> List[MathematicalProblem]:
        """Load and process GSM8K dataset."""
        cache_file = self.cache_dir / f"gsm8k_{split}_processed.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached GSM8K dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        try:
            # Load from HuggingFace
            dataset = load_dataset("gsm8k", "main", split=split)
            
            problems = []
            for item in dataset:
                problem = MathematicalProblem(
                    problem=item["question"],
                    solution=item["answer"],
                    final_answer=self._extract_final_answer(item["answer"]),
                    difficulty_level="elementary",
                    problem_type="arithmetic_word_problem",
                    source_dataset="gsm8k",
                    mathematical_concepts=self._extract_mathematical_concepts(item["question"]),
                    token_count=len(item["question"].split()) + len(item["answer"].split())
                )
                
                # Enhance with SOTA techniques
                problem = self._enhance_with_sota_techniques(problem)
                problems.append(problem)
            
            # Cache processed dataset
            with open(cache_file, 'wb') as f:
                pickle.dump(problems, f)
            
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            return self._load_local_gsm8k_dataset(split)
    
    def _load_local_math_dataset(self, split: str) -> List[MathematicalProblem]:
        """Load MATH dataset from local files as fallback."""
        local_file = Path(f"local_data/math_{split}.jsonl")
        
        if not local_file.exists():
            logger.warning(f"Local MATH dataset file not found: {local_file}")
            return []
        
        problems = []
        with open(local_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                problem = MathematicalProblem(
                    problem=item["problem"],
                    solution=item["solution"],
                    final_answer=self._extract_final_answer(item["solution"]),
                    difficulty_level=item.get("level", "unknown"),
                    problem_type=item.get("type", "general"),
                    source_dataset="math_local",
                    mathematical_concepts=self._extract_mathematical_concepts(item["problem"]),
                    token_count=len(item["problem"].split()) + len(item["solution"].split())
                )
                problem = self._enhance_with_sota_techniques(problem)
                problems.append(problem)
        
        return problems
    
    def _load_local_gsm8k_dataset(self, split: str) -> List[MathematicalProblem]:
        """Load GSM8K dataset from local files as fallback."""
        local_file = Path(f"local_data/gsm8k_{split}.jsonl")
        
        if not local_file.exists():
            logger.warning(f"Local GSM8K dataset file not found: {local_file}")
            return []
        
        problems = []
        with open(local_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                problem = MathematicalProblem(
                    problem=item["question"],
                    solution=item["answer"],
                    final_answer=self._extract_final_answer(item["answer"]),
                    difficulty_level="elementary",
                    problem_type="arithmetic_word_problem",
                    source_dataset="gsm8k_local",
                    mathematical_concepts=self._extract_mathematical_concepts(item["question"]),
                    token_count=len(item["question"].split()) + len(item["answer"].split())
                )
                problem = self._enhance_with_sota_techniques(problem)
                problems.append(problem)
        
        return problems
    
    def load_sota_specific_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load SOTA method-specific datasets."""
        logger.info(f"Loading SOTA-specific datasets for method: {self.sota_method}")
        
        sota_datasets = {}
        
        if self.sota_method == "deepseekmath":
            sota_datasets.update(self._load_deepseekmath_datasets())
        elif self.sota_method == "orca_math":
            sota_datasets.update(self._load_orca_math_datasets())
        elif self.sota_method == "dotamath":
            sota_datasets.update(self._load_dotamath_datasets())
        elif self.sota_method == "internlm_math":
            sota_datasets.update(self._load_internlm_datasets())
        elif self.sota_method == "mindstar":
            sota_datasets.update(self._load_mindstar_datasets())
        
        return sota_datasets
    
    def _load_deepseekmath_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load DeepSeekMath-specific datasets and generate continued pretraining data."""
        datasets = {}
        
        # Simulate DeepSeekMath continued pretraining corpus
        logger.info("Generating DeepSeekMath-style mathematical corpus...")
        datasets["deepseek_math_corpus"] = self._generate_deepseek_corpus()
        
        # Mathematical textbook-style problems
        datasets["deepseek_textbook"] = self._generate_textbook_problems()
        
        return datasets
    
    def _load_orca_math_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load Orca-Math-specific datasets with multi-agent generation."""
        datasets = {}
        
        # Multi-agent synthetic data generation
        logger.info("Generating Orca-Math-style multi-agent synthetic data...")
        datasets["orca_math_synthetic"] = self._generate_multi_agent_problems()
        
        # Iterative preference learning data
        datasets["orca_preference"] = self._generate_preference_learning_data()
        
        return datasets
    
    def _load_dotamath_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load DotaMath-specific datasets with decomposition reasoning."""
        datasets = {}
        
        # Decomposition of thought problems
        logger.info("Generating DotaMath-style decomposition problems...")
        datasets["dotamath_decomposition"] = self._generate_decomposition_problems()
        
        # Code-assisted mathematical reasoning
        datasets["dotamath_code"] = self._generate_code_assisted_problems()
        
        return datasets
    
    def _load_internlm_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load InternLM-Math-specific datasets with formal reasoning."""
        datasets = {}
        
        # Formal mathematical proofs
        logger.info("Generating InternLM-Math-style formal reasoning data...")
        datasets["internlm_math_formal"] = self._generate_formal_proof_problems()
        
        # LEAN-integrated problems
        datasets["internlm_lean"] = self._generate_lean_problems()
        
        return datasets
    
    def _load_mindstar_datasets(self) -> Dict[str, List[MathematicalProblem]]:
        """Load MindStar-specific datasets for inference optimization."""
        datasets = {}
        
        # Inference-optimized problems
        logger.info("Generating MindStar-style inference-optimized data...")
        datasets["mindstar_inference"] = self._generate_inference_optimized_problems()
        
        return datasets
    
    def _enhance_with_sota_techniques(self, problem: MathematicalProblem) -> MathematicalProblem:
        """Enhance problem with SOTA-specific techniques."""
        
        # Chain-of-thought enhancement
        if self.use_chain_of_thought:
            problem.chain_of_thought = self._generate_chain_of_thought(problem)
        
        # Code assistance enhancement
        if self.use_code_assistance or self.current_config.get("code_assistance", False):
            problem.code_solution = self._generate_code_solution(problem)
            problem.verification_code = self._generate_verification_code(problem)
            problem.requires_code = True
        
        # Multi-agent enhancement
        if self.use_multi_agent_learning or self.current_config.get("multi_agent_generation", False):
            problem.multi_agent_variants = self._generate_multi_agent_variants(problem)
        
        # Decomposition enhancement
        if self.current_config.get("decomposition_reasoning", False):
            problem.decomposition_steps = self._generate_decomposition_steps(problem)
        
        # Formal reasoning enhancement
        if self.current_config.get("formal_reasoning", False):
            problem.formal_proof = self._generate_formal_proof(problem)
        
        # Estimate difficulty
        problem.estimated_difficulty = self._estimate_difficulty(problem)
        
        return problem
    
    def _generate_chain_of_thought(self, problem: MathematicalProblem) -> str:
        """Generate chain-of-thought reasoning for the problem."""
        # Simulate CoT generation based on problem and solution
        steps = problem.solution.split('\n')
        cot_steps = []
        
        for i, step in enumerate(steps):
            if step.strip():
                cot_step = f"Step {i+1}: Let me think about this. {step.strip()}"
                cot_steps.append(cot_step)
        
        return "\n".join(cot_steps)
    
    def _generate_code_solution(self, problem: MathematicalProblem) -> str:
        """Generate Python code solution for the problem."""
        # Simplified code generation based on problem type
        if "calculate" in problem.problem.lower() or "compute" in problem.problem.lower():
            return f"""
# Mathematical computation for: {problem.problem[:50]}...
import math
import sympy as sp

def solve_problem():
    # Extract numerical values and perform calculation
    result = None
    # Computation logic would go here
    return result

answer = solve_problem()
print(f"Final answer: {{answer}}")
"""
        return ""
    
    def _generate_verification_code(self, problem: MathematicalProblem) -> str:
        """Generate verification code for the solution."""
        return f"""
# Verification for problem: {problem.problem[:50]}...
def verify_solution(answer):
    # Verification logic
    expected = {problem.final_answer}
    return abs(float(answer) - float(expected)) < 1e-6

# Test verification
is_correct = verify_solution({problem.final_answer})
print(f"Solution verified: {{is_correct}}")
"""
    
    def _generate_multi_agent_variants(self, problem: MathematicalProblem) -> List[str]:
        """Generate multiple agent perspectives on the problem."""
        variants = []
        
        # Agent 1: Step-by-step approach
        variants.append(f"Agent 1 (Methodical): Let me solve this step by step.\n{problem.solution}")
        
        # Agent 2: Conceptual approach
        variants.append(f"Agent 2 (Conceptual): Let me think about the underlying concepts first.\n{problem.solution}")
        
        # Agent 3: Alternative method
        variants.append(f"Agent 3 (Alternative): Here's another way to approach this problem.\n{problem.solution}")
        
        return variants
    
    def _generate_decomposition_steps(self, problem: MathematicalProblem) -> List[str]:
        """Generate decomposition of thought steps."""
        # Decompose the problem into sub-problems
        steps = [
            f"1. Understanding: What is the problem asking for?",
            f"2. Information: What information do we have?",
            f"3. Strategy: What approach should we use?",
            f"4. Execution: Let's solve step by step.",
            f"5. Verification: Does our answer make sense?"
        ]
        return steps
    
    def _generate_formal_proof(self, problem: MathematicalProblem) -> str:
        """Generate formal mathematical proof structure."""
        return f"""
Theorem: {problem.problem}

Proof:
Given: [Problem conditions]
To Prove: [Target conclusion]

Proof by [method]:
1. [Premise 1]
2. [Logical step 1]
3. [Logical step 2]
...
Therefore, [conclusion] ∎
"""
    
    def _estimate_difficulty(self, problem: MathematicalProblem) -> float:
        """Estimate problem difficulty based on various factors."""
        difficulty = 0.5  # Base difficulty
        
        # Token count factor
        if problem.token_count > 200:
            difficulty += 0.2
        elif problem.token_count > 100:
            difficulty += 0.1
        
        # Mathematical complexity
        math_indicators = ["integral", "derivative", "matrix", "probability", "combinatorics"]
        for indicator in math_indicators:
            if indicator in problem.problem.lower():
                difficulty += 0.1
                break
        
        # Code requirement factor
        if problem.requires_code:
            difficulty += 0.15
        
        return min(difficulty, 1.0)
    
    def _extract_final_answer(self, solution: str) -> str:
        """Extract final numerical answer from solution."""
        # Look for various answer patterns
        patterns = [
            r'####\s*([0-9,]+(?:\.[0-9]+)?)',  # GSM8K style
            r'\\boxed\{([^}]+)\}',             # LaTeX boxed
            r'(?:answer|result|solution)(?:\s*is)?:?\s*([0-9,]+(?:\.[0-9]+)?)',  # Natural language
            r'\$([0-9,]+(?:\.[0-9]+)?)\$',     # Dollar signs
            r'\b([0-9,]+(?:\.[0-9]+)?)\s*(?:\.|$)'  # Last number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        
        return ""
    
    def _extract_mathematical_concepts(self, problem_text: str) -> List[str]:
        """Extract mathematical concepts from problem text."""
        concepts = []
        
        concept_keywords = {
            "algebra": ["equation", "variable", "solve", "polynomial"],
            "geometry": ["triangle", "circle", "angle", "area", "perimeter"],
            "calculus": ["derivative", "integral", "limit", "function"],
            "probability": ["probability", "random", "expected", "distribution"],
            "number_theory": ["prime", "divisible", "gcd", "modular"],
            "combinatorics": ["permutation", "combination", "counting"]
        }
        
        problem_lower = problem_text.lower()
        for concept, keywords in concept_keywords.items():
            if any(keyword in problem_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts or ["general"]
    
    def prepare_training_dataset(self, dataset_names: List[str]) -> 'SOTAMathDataset':
        """Prepare combined training dataset from specified sources."""
        logger.info(f"Preparing training dataset from: {dataset_names}")
        
        all_problems = []
        
        # Load base datasets
        base_datasets = self.load_base_datasets()
        for name in dataset_names:
            if name in base_datasets:
                all_problems.extend(base_datasets[name])
        
        # Load SOTA-specific datasets
        sota_datasets = self.load_sota_specific_datasets()
        for name in dataset_names:
            if name in sota_datasets:
                all_problems.extend(sota_datasets[name])
        
        # Apply SOTA-specific filtering and augmentation
        all_problems = self._apply_sota_filtering(all_problems)
        all_problems = self._apply_sota_augmentation(all_problems)
        
        # Shuffle and create dataset
        random.shuffle(all_problems)
        
        logger.info(f"Prepared training dataset with {len(all_problems)} problems")
        return SOTAMathDataset(all_problems, self.tokenizer, self.max_length, is_training=True)
    
    def prepare_evaluation_dataset(self, dataset_names: List[str]) -> 'SOTAMathDataset':
        """Prepare evaluation dataset from specified sources."""
        logger.info(f"Preparing evaluation dataset from: {dataset_names}")
        
        all_problems = []
        
        # Load base datasets (test splits)
        for name in dataset_names:
            if name == "math_test":
                problems = self._load_math_dataset("test")
                all_problems.extend(problems[:1000])  # Limit for efficiency
            elif name == "gsm8k_test":
                problems = self._load_gsm8k_dataset("test")
                all_problems.extend(problems[:500])   # Limit for efficiency
            elif name == "aime":
                problems = self._generate_aime_problems()
                all_problems.extend(problems)
        
        logger.info(f"Prepared evaluation dataset with {len(all_problems)} problems")
        return SOTAMathDataset(all_problems, self.tokenizer, self.max_length, is_training=False)
    
    def _apply_sota_filtering(self, problems: List[MathematicalProblem]) -> List[MathematicalProblem]:
        """Apply SOTA-specific filtering to problems."""
        if self.sota_method == "deepseekmath":
            # Filter for mathematical complexity
            return [p for p in problems if p.estimated_difficulty >= 0.4]
        elif self.sota_method == "orca_math":
            # Prefer problems suitable for multi-agent learning
            return [p for p in problems if len(p.multi_agent_variants or []) > 0]
        elif self.sota_method == "dotamath":
            # Focus on decomposable problems
            return [p for p in problems if p.decomposition_steps is not None]
        else:
            return problems
    
    def _apply_sota_augmentation(self, problems: List[MathematicalProblem]) -> List[MathematicalProblem]:
        """Apply SOTA-specific data augmentation."""
        augmented = problems.copy()
        
        if self.sota_method == "orca_math" and self.current_config.get("multi_agent_generation", False):
            # Generate additional problems using multi-agent approach
            logger.info("Applying Orca-Math multi-agent augmentation...")
            additional_problems = self._generate_additional_multi_agent_problems(problems[:100])
            augmented.extend(additional_problems)
        
        return augmented
    
    # Placeholder methods for generating SOTA-specific datasets
    def _generate_deepseek_corpus(self) -> List[MathematicalProblem]:
        """Generate DeepSeekMath-style mathematical corpus."""
        return []  # Placeholder
    
    def _generate_textbook_problems(self) -> List[MathematicalProblem]:
        """Generate textbook-style mathematical problems."""
        return []  # Placeholder
    
    def _generate_multi_agent_problems(self) -> List[MathematicalProblem]:
        """Generate multi-agent mathematical problems."""
        return []  # Placeholder
    
    def _generate_preference_learning_data(self) -> List[MathematicalProblem]:
        """Generate preference learning data."""
        return []  # Placeholder
    
    def _generate_decomposition_problems(self) -> List[MathematicalProblem]:
        """Generate decomposition of thought problems."""
        return []  # Placeholder
    
    def _generate_code_assisted_problems(self) -> List[MathematicalProblem]:
        """Generate code-assisted mathematical problems."""
        return []  # Placeholder
    
    def _generate_formal_proof_problems(self) -> List[MathematicalProblem]:
        """Generate formal proof problems."""
        return []  # Placeholder
    
    def _generate_lean_problems(self) -> List[MathematicalProblem]:
        """Generate LEAN-integrated problems."""
        return []  # Placeholder
    
    def _generate_inference_optimized_problems(self) -> List[MathematicalProblem]:
        """Generate inference-optimized problems."""
        return []  # Placeholder
    
    def _generate_aime_problems(self) -> List[MathematicalProblem]:
        """Generate AIME-style problems for evaluation."""
        return []  # Placeholder
    
    def _generate_additional_multi_agent_problems(self, base_problems: List[MathematicalProblem]) -> List[MathematicalProblem]:
        """Generate additional problems using multi-agent approach."""
        return []  # Placeholder

class SOTAMathDataset(Dataset):
    """PyTorch Dataset for SOTA mathematical reasoning with enhanced preprocessing."""
    
    def __init__(
        self,
        problems: List[MathematicalProblem],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        is_training: bool = True
    ):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        
        # Preprocess problems for efficient training
        self.processed_problems = self._preprocess_problems()
        
    def _preprocess_problems(self) -> List[Dict[str, Any]]:
        """Preprocess problems for training/evaluation."""
        processed = []
        
        for problem in self.problems:
            # Create instruction-following format
            if self.is_training:
                instruction = self._create_training_instruction(problem)
            else:
                instruction = self._create_evaluation_instruction(problem)
            
            processed.append({
                'instruction': instruction,
                'problem_obj': problem
            })
        
        return processed
    
    def _create_training_instruction(self, problem: MathematicalProblem) -> str:
        """Create training instruction with SOTA techniques."""
        instruction_parts = []
        
        # Base instruction
        instruction_parts.append("You are a mathematical reasoning expert. Solve the following problem step by step.")
        
        # Add SOTA-specific instructions
        if problem.chain_of_thought:
            instruction_parts.append("Use chain-of-thought reasoning to work through the problem.")
        
        if problem.requires_code:
            instruction_parts.append("You may use Python code to assist with calculations.")
        
        if problem.decomposition_steps:
            instruction_parts.append("Break down the problem into smaller sub-problems.")
        
        # Problem and solution
        instruction_parts.extend([
            f"\nProblem: {problem.problem}",
            f"\nSolution: {problem.solution}"
        ])
        
        return "\n".join(instruction_parts)
    
    def _create_evaluation_instruction(self, problem: MathematicalProblem) -> str:
        """Create evaluation instruction for testing."""
        return f"Solve this mathematical problem step by step:\n\nProblem: {problem.problem}\n\nSolution:"
    
    def __len__(self):
        return len(self.processed_problems)
    
    def __getitem__(self, idx):
        item = self.processed_problems[idx]
        instruction = item['instruction']
        problem_obj = item['problem_obj']
        
        # Tokenize for training
        if self.is_training:
            # For training, we want the full instruction + solution
            encoding = self.tokenizer(
                instruction,
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            
            # Labels are the same as input_ids for causal LM
            labels = input_ids.clone()
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'problem_obj': problem_obj
            }
        else:
            # For evaluation, we only need the problem part
            problem_text = f"Solve this mathematical problem step by step:\n\nProblem: {problem_obj.problem}\n\nSolution:"
            
            encoding = self.tokenizer(
                problem_text,
                truncation=True,
                padding=False,
                max_length=self.max_length // 2,  # Leave space for generation
                return_tensors="pt"
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'problem_obj': problem_obj
            }

# Example usage and testing
if __name__ == "__main__":
    # Test the SOTA dataset loader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test different SOTA methods
    for method in ["deepseekmath", "orca_math", "dotamath", "internlm_math", "mindstar"]:
        print(f"\nTesting {method} method:")
        
        loader = SOTAMathDatasetLoader(
            tokenizer=tokenizer,
            sota_method=method,
            use_chain_of_thought=True,
            use_code_assistance=(method in ["dotamath", "internlm_math"]),
            max_length=2048
        )
        
        # Load datasets
        base_datasets = loader.load_base_datasets()
        sota_datasets = loader.load_sota_specific_datasets()
        
        print(f"Base datasets: {list(base_datasets.keys())}")
        print(f"SOTA datasets: {list(sota_datasets.keys())}")
        
        # Create training dataset
        if base_datasets.get("math") or base_datasets.get("gsm8k"):
            train_dataset = loader.prepare_training_dataset(["math", "gsm8k"])
            print(f"Training dataset size: {len(train_dataset)}")
            
            # Test dataset item
            if len(train_dataset) > 0:
                sample = train_dataset[0]
                print(f"Sample input shape: {sample['input_ids'].shape}")
                print(f"Problem type: {sample['problem_obj'].problem_type}")
                print(f"Estimated difficulty: {sample['problem_obj'].estimated_difficulty}")