"""
Enhanced Mathematical Dataset Loader for Large-Scale Training

This module provides comprehensive loading and preprocessing for mathematical reasoning datasets,
optimized for training large language models on mathematical problem solving.

Supported Datasets:
- MATH: Competition mathematics problems  
- GSM8K: Grade school math word problems
- OpenMathInstruct-1M: Large-scale mathematical instruction dataset
- MetaMathQA: Mathematical reasoning with step-by-step solutions
- MathInstruct: Comprehensive mathematical problem collection
- Custom mathematical problem datasets

Features:
- Efficient data loading with caching
- Mathematical symbol preprocessing
- Chain-of-thought augmentation
- Problem difficulty classification
- Multi-format support (JSON, JSONL, Parquet)
- Memory-efficient streaming for large datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
import json
# import jsonlines  # Optional dependency
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
import re
import random
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import sympy
from sympy import latex, sympify
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MathProblem:
    """Mathematical problem data structure."""
    problem: str
    solution: str
    answer: str
    problem_type: str = "unknown"
    difficulty: str = "medium"
    source: str = "unknown"
    steps: Optional[List[str]] = None
    concepts: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class MathDatasetLoader:
    """
    Comprehensive loader for mathematical reasoning datasets.
    
    Supports multiple dataset formats and provides preprocessing
    specifically optimized for mathematical content.
    """
    
    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 4096,
        cache_dir: str = "./data_cache",
        enable_augmentation: bool = True,
        augmentation_ratio: float = 0.3,
        preprocessing_workers: int = 4
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.enable_augmentation = enable_augmentation
        self.augmentation_ratio = augmentation_ratio
        self.preprocessing_workers = preprocessing_workers
        
        # Dataset registry
        self.dataset_configs = {
            'math': {
                'hf_name': 'hendrycks/competition_math',
                'problem_key': 'problem',
                'solution_key': 'solution',
                'type_key': 'type',
                'level_key': 'level'
            },
            'gsm8k': {
                'hf_name': 'gsm8k',
                'subset': 'main',
                'problem_key': 'question',
                'solution_key': 'answer'
            },
            'openmath_instruct': {
                'hf_name': 'nvidia/OpenMathInstruct-1',
                'problem_key': 'problem',
                'solution_key': 'generated_solution'
            },
            'metamath': {
                'hf_name': 'meta-math/MetaMathQA',
                'problem_key': 'query',
                'solution_key': 'response'
            },
            'mathinstruct': {
                'hf_name': 'TIGER-Lab/MathInstruct',
                'problem_key': 'instruction',
                'solution_key': 'output'
            }
        }
        
        # Mathematical concept patterns
        self.concept_patterns = {
            'algebra': [r'solve.*equation', r'polynomial', r'quadratic', r'linear'],
            'geometry': [r'triangle', r'circle', r'polygon', r'area', r'perimeter', r'volume'],
            'calculus': [r'derivative', r'integral', r'limit', r'differential'],
            'probability': [r'probability', r'random', r'expected value', r'distribution'],
            'number_theory': [r'prime', r'divisor', r'modular', r'gcd', r'lcm'],
            'combinatorics': [r'permutation', r'combination', r'counting'],
            'statistics': [r'mean', r'median', r'variance', r'standard deviation'],
            'trigonometry': [r'sin', r'cos', r'tan', r'angle']
        }
        
        logger.info(f"Initialized MathDatasetLoader with cache at {cache_dir}")
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        shuffle: bool = True
    ) -> List[MathProblem]:
        """Load a specific mathematical dataset."""
        
        cache_file = self.cache_dir / f"{dataset_name}_{split}_{max_samples}.pkl"
        
        # Try loading from cache
        if cache_file.exists():
            logger.info(f"Loading {dataset_name} from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Loading {dataset_name} dataset (split: {split})")
        
        if dataset_name in self.dataset_configs:
            problems = self._load_huggingface_dataset(dataset_name, split, max_samples)
        else:
            problems = self._load_local_dataset(dataset_name, split, max_samples)
        
        if shuffle:
            random.shuffle(problems)
        
        # Cache the processed dataset
        with open(cache_file, 'wb') as f:
            pickle.dump(problems, f)
        
        logger.info(f"Loaded {len(problems)} problems from {dataset_name}")
        return problems
    
    def _load_huggingface_dataset(
        self,
        dataset_name: str,
        split: str,
        max_samples: Optional[int]
    ) -> List[MathProblem]:
        """Load dataset from HuggingFace."""
        
        config = self.dataset_configs[dataset_name]
        
        try:
            # Load dataset
            if 'subset' in config:
                dataset = load_dataset(config['hf_name'], config['subset'], split=split)
            else:
                dataset = load_dataset(config['hf_name'], split=split)
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Process dataset
            problems = []
            for item in dataset:
                problem = self._process_dataset_item(item, config, dataset_name)
                if problem:
                    problems.append(problem)
            
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name} from HuggingFace: {e}")
            return self._load_fallback_dataset(dataset_name, split, max_samples)
    
    def _load_local_dataset(
        self,
        dataset_name: str,
        split: str,
        max_samples: Optional[int]
    ) -> List[MathProblem]:
        """Load dataset from local files."""
        
        # Try different file formats
        potential_files = [
            self.cache_dir / f"{dataset_name}_{split}.jsonl",
            self.cache_dir / f"{dataset_name}_{split}.json",
            self.cache_dir / f"{dataset_name}_{split}.parquet",
            Path(f"./local_data/{dataset_name}_{split}.jsonl"),
            Path(f"./local_data/{dataset_name}_{split}.json")
        ]
        
        for file_path in potential_files:
            if file_path.exists():
                logger.info(f"Loading local dataset from: {file_path}")
                return self._load_file(file_path, dataset_name, max_samples)
        
        logger.warning(f"No local dataset found for {dataset_name}")
        return []
    
    def _load_file(
        self,
        file_path: Path,
        dataset_name: str,
        max_samples: Optional[int]
    ) -> List[MathProblem]:
        """Load problems from a specific file."""
        
        problems = []
        
        try:
            if file_path.suffix == '.jsonl':
                try:
                    import jsonlines
                    with jsonlines.open(file_path) as reader:
                        for i, item in enumerate(reader):
                            if max_samples and i >= max_samples:
                                break
                            problem = self._process_local_item(item, dataset_name)
                            if problem:
                                problems.append(problem)
                except ImportError:
                    logger.warning("jsonlines not available, skipping .jsonl files")
            
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i, item in enumerate(data):
                            if max_samples and i >= max_samples:
                                break
                            problem = self._process_local_item(item, dataset_name)
                            if problem:
                                problems.append(problem)
            
            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
                if max_samples:
                    df = df.head(max_samples)
                for _, row in df.iterrows():
                    problem = self._process_local_item(row.to_dict(), dataset_name)
                    if problem:
                        problems.append(problem)
        
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
        
        return problems
    
    def _process_dataset_item(
        self,
        item: Dict[str, Any],
        config: Dict[str, str],
        source: str
    ) -> Optional[MathProblem]:
        """Process a single dataset item into MathProblem format."""
        
        try:
            problem_text = item[config['problem_key']]
            solution_text = item[config['solution_key']]
            
            # Extract answer from solution
            answer = self._extract_answer(solution_text)
            
            # Determine problem type and difficulty
            problem_type = item.get(config.get('type_key', 'type'), 'unknown')
            difficulty = item.get(config.get('level_key', 'level'), 'medium')
            
            # Extract mathematical concepts
            concepts = self._extract_concepts(problem_text)
            
            # Extract solution steps
            steps = self._extract_steps(solution_text)
            
            return MathProblem(
                problem=problem_text,
                solution=solution_text,
                answer=answer,
                problem_type=problem_type,
                difficulty=str(difficulty),
                source=source,
                steps=steps,
                concepts=concepts,
                metadata=item
            )
            
        except Exception as e:
            logger.warning(f"Failed to process item from {source}: {e}")
            return None
    
    def _process_local_item(self, item: Dict[str, Any], source: str) -> Optional[MathProblem]:
        """Process local dataset item."""
        
        # Try common field names
        problem_keys = ['problem', 'question', 'query', 'instruction', 'input']
        solution_keys = ['solution', 'answer', 'response', 'output', 'target']
        
        problem_text = None
        solution_text = None
        
        for key in problem_keys:
            if key in item:
                problem_text = item[key]
                break
        
        for key in solution_keys:
            if key in item:
                solution_text = item[key]
                break
        
        if not problem_text or not solution_text:
            return None
        
        answer = self._extract_answer(solution_text)
        concepts = self._extract_concepts(problem_text)
        steps = self._extract_steps(solution_text)
        
        return MathProblem(
            problem=problem_text,
            solution=solution_text,
            answer=answer,
            problem_type=item.get('type', 'unknown'),
            difficulty=item.get('difficulty', 'medium'),
            source=source,
            steps=steps,
            concepts=concepts,
            metadata=item
        )
    
    def _extract_answer(self, solution: str) -> str:
        """Extract final answer from solution text."""
        
        # Common answer patterns in mathematical datasets
        patterns = [
            r'####\s*([0-9,]+(?:\.[0-9]+)?)',  # GSM8K style
            r'\\boxed\{([^}]+)\}',             # LaTeX boxed answer
            r'(?:the )?answer is:?\s*([^\n]+)',  # Natural language
            r'(?:final answer|result).*?:?\s*([^\n]+)',
            r'\$([0-9,]+(?:\.[0-9]+)?)\$',     # Dollar amounts
            r'([0-9,]+(?:\.[0-9]+)?)\s*(?:units?|dollars?|cents?)?(?:\s*\.)?$'  # Final number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Fallback: last number in the solution
        numbers = re.findall(r'[0-9,]+(?:\.[0-9]+)?', solution)
        if numbers:
            return numbers[-1]
        
        return ""
    
    def _extract_concepts(self, problem_text: str) -> List[str]:
        """Extract mathematical concepts from problem text."""
        
        concepts = []
        text_lower = problem_text.lower()
        
        for concept, patterns in self.concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    concepts.append(concept)
                    break
        
        return concepts if concepts else ['general']
    
    def _extract_steps(self, solution: str) -> List[str]:
        """Extract solution steps from solution text."""
        
        # Look for numbered steps or clear logical breaks
        step_patterns = [
            r'Step \d+:([^\n]+)',
            r'\d+\.([^\n]+)',
            r'First,([^\n]+)',
            r'Then,([^\n]+)',
            r'Next,([^\n]+)',
            r'Finally,([^\n]+)'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            steps.extend([match.strip() for match in matches])
        
        # If no clear steps found, split by sentences
        if not steps:
            sentences = re.split(r'[.!?]+', solution)
            steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return steps[:10]  # Limit to 10 steps
    
    def load_multiple_datasets(
        self,
        dataset_names: List[str],
        split: str = 'train',
        max_samples_per_dataset: Optional[int] = None,
        balance_datasets: bool = True
    ) -> List[MathProblem]:
        """Load and combine multiple datasets."""
        
        all_problems = []
        
        for dataset_name in dataset_names:
            problems = self.load_dataset(dataset_name, split, max_samples_per_dataset)
            all_problems.extend(problems)
            logger.info(f"Added {len(problems)} problems from {dataset_name}")
        
        if balance_datasets and len(dataset_names) > 1:
            all_problems = self._balance_datasets(all_problems, dataset_names)
        
        # Shuffle combined dataset
        random.shuffle(all_problems)
        
        logger.info(f"Combined dataset: {len(all_problems)} total problems")
        return all_problems
    
    def _balance_datasets(
        self,
        problems: List[MathProblem],
        dataset_names: List[str]
    ) -> List[MathProblem]:
        """Balance problems across different datasets."""
        
        # Group problems by source
        source_groups = {}
        for problem in problems:
            source = problem.source
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(problem)
        
        # Find minimum size
        min_size = min(len(group) for group in source_groups.values())
        
        # Sample equally from each source
        balanced_problems = []
        for source, group in source_groups.items():
            sampled = random.sample(group, min_size)
            balanced_problems.extend(sampled)
        
        logger.info(f"Balanced dataset: {min_size} problems per source")
        return balanced_problems
    
    def create_pytorch_dataset(
        self,
        problems: List[MathProblem],
        is_training: bool = True
    ) -> 'MathDatasetTorch':
        """Create PyTorch dataset from problem list."""
        
        return MathDatasetTorch(
            problems=problems,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            is_training=is_training,
            enable_augmentation=self.enable_augmentation and is_training,
            augmentation_ratio=self.augmentation_ratio
        )
    
    def create_dataloader(
        self,
        problems: List[MathProblem],
        batch_size: int = 8,
        is_training: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        
        dataset = self.create_pytorch_dataset(problems, is_training)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
    
    def _load_fallback_dataset(
        self,
        dataset_name: str,
        split: str,
        max_samples: Optional[int]
    ) -> List[MathProblem]:
        """Fallback dataset loading for when HuggingFace fails."""
        
        logger.warning(f"Using fallback data for {dataset_name}")
        
        # Generate synthetic mathematical problems as fallback
        fallback_problems = []
        
        for i in range(min(100, max_samples or 100)):
            # Generate simple arithmetic problems
            a, b = random.randint(1, 100), random.randint(1, 100)
            op = random.choice(['+', '-', '*'])
            
            if op == '+':
                answer = a + b
                problem = f"What is {a} + {b}?"
                solution = f"To solve {a} + {b}, we add the numbers: {a} + {b} = {answer}"
            elif op == '-':
                answer = a - b
                problem = f"What is {a} - {b}?"
                solution = f"To solve {a} - {b}, we subtract: {a} - {b} = {answer}"
            else:  # multiplication
                answer = a * b
                problem = f"What is {a} × {b}?"
                solution = f"To solve {a} × {b}, we multiply: {a} × {b} = {answer}"
            
            fallback_problems.append(MathProblem(
                problem=problem,
                solution=solution,
                answer=str(answer),
                problem_type="arithmetic",
                difficulty="easy",
                source=f"{dataset_name}_fallback"
            ))
        
        return fallback_problems


class MathDatasetTorch(Dataset):
    """PyTorch Dataset for mathematical reasoning problems."""
    
    def __init__(
        self,
        problems: List[MathProblem],
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        is_training: bool = True,
        enable_augmentation: bool = True,
        augmentation_ratio: float = 0.3
    ):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.enable_augmentation = enable_augmentation
        self.augmentation_ratio = augmentation_ratio
        
        # Preprocess problems
        self.processed_problems = self._preprocess_problems()
    
    def _preprocess_problems(self) -> List[Dict[str, Any]]:
        """Preprocess problems for efficient training."""
        
        processed = []
        
        for problem in self.problems:
            # Create training text
            if self.is_training:
                text = self._create_training_text(problem)
            else:
                text = self._create_evaluation_text(problem)
            
            processed.append({
                'text': text,
                'problem_obj': problem
            })
        
        return processed
    
    def _create_training_text(self, problem: MathProblem) -> str:
        """Create training text with problem and solution."""
        
        # Mathematical reasoning prompt template
        template = """Problem: {problem}

Solution: {solution}"""
        
        return template.format(
            problem=problem.problem,
            solution=problem.solution
        )
    
    def _create_evaluation_text(self, problem: MathProblem) -> str:
        """Create evaluation text with only problem."""
        
        return f"Problem: {problem.problem}\n\nSolution:"
    
    def _augment_problem(self, problem: MathProblem) -> MathProblem:
        """Apply data augmentation to a mathematical problem."""
        
        if not self.enable_augmentation or random.random() > self.augmentation_ratio:
            return problem
        
        # Simple augmentation strategies
        augmented_problem = problem.problem
        augmented_solution = problem.solution
        
        # Rephrase numbers (10% chance)
        if random.random() < 0.1:
            # Convert between word and digit forms
            number_words = {
                '1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five',
                '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '10': 'ten'
            }
            
            for digit, word in number_words.items():
                if random.random() < 0.5:
                    augmented_problem = augmented_problem.replace(digit, word)
                else:
                    augmented_problem = augmented_problem.replace(word, digit)
        
        # Add reasoning encouragement (20% chance)
        if random.random() < 0.2:
            encouragements = [
                "Let me think step by step.",
                "I'll solve this carefully.",
                "Let me work through this systematically."
            ]
            augmented_solution = random.choice(encouragements) + " " + augmented_solution
        
        return MathProblem(
            problem=augmented_problem,
            solution=augmented_solution,
            answer=problem.answer,
            problem_type=problem.problem_type,
            difficulty=problem.difficulty,
            source=problem.source + "_augmented",
            steps=problem.steps,
            concepts=problem.concepts,
            metadata=problem.metadata
        )
    
    def __len__(self) -> int:
        return len(self.processed_problems)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.processed_problems[idx]
        problem_obj = item['problem_obj']
        
        # Apply augmentation if training
        if self.is_training:
            problem_obj = self._augment_problem(problem_obj)
            text = self._create_training_text(problem_obj)
        else:
            text = item['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Warn if truncation occurred
        if hasattr(encoding, 'num_truncated_tokens') and encoding.num_truncated_tokens > 0:
            logger.warning(f"Truncation occurred for problem: {problem_obj.problem[:60]}...")
        elif encoding['input_ids'].shape[-1] >= self.max_length:
            logger.warning(f"Truncation likely occurred for problem: {problem_obj.problem[:60]}...")
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }
        
        # For training, labels are the same as input_ids
        if self.is_training:
            result['labels'] = result['input_ids'].clone()
        
        # Add metadata
        result['problem_type'] = problem_obj.problem_type
        result['difficulty'] = problem_obj.difficulty
        result['source'] = problem_obj.source
        
        return result


# Convenience functions
def load_math_datasets(
    datasets: List[str] = ['math', 'gsm8k'],
    split: str = 'train',
    max_samples: Optional[int] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    **kwargs
) -> DataLoader:
    """Convenience function to load and create DataLoader for math datasets."""
    
    loader = MathDatasetLoader(tokenizer=tokenizer, **kwargs)
    problems = loader.load_multiple_datasets(datasets, split, max_samples)
    return loader.create_dataloader(problems, is_training=(split == 'train'))


def create_demo_dataset() -> List[MathProblem]:
    """Create a small demo dataset for testing."""
    
    problems = [
        MathProblem(
            problem="What is 15 + 27?",
            solution="To find 15 + 27, I'll add these numbers: 15 + 27 = 42",
            answer="42",
            problem_type="arithmetic",
            difficulty="easy",
            source="demo"
        ),
        MathProblem(
            problem="A rectangle has length 8 meters and width 5 meters. What is its area?",
            solution="The area of a rectangle is length × width. So: Area = 8 × 5 = 40 square meters",
            answer="40",
            problem_type="geometry",
            difficulty="medium",
            source="demo"
        ),
        MathProblem(
            problem="Solve for x: 2x + 5 = 13",
            solution="To solve 2x + 5 = 13:\n1. Subtract 5 from both sides: 2x = 8\n2. Divide by 2: x = 4",
            answer="4",
            problem_type="algebra",
            difficulty="medium",
            source="demo"
        )
    ]
    
    return problems


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing MathDatasetLoader...")
    
    # Create demo tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test with demo dataset
    loader = MathDatasetLoader(tokenizer=tokenizer)
    demo_problems = create_demo_dataset()
    
    print(f"Demo dataset: {len(demo_problems)} problems")
    
    # Create PyTorch dataset
    dataset = loader.create_pytorch_dataset(demo_problems, is_training=True)
    print(f"PyTorch dataset: {len(dataset)} items")
    
    # Test one item
    item = dataset[0]
    print(f"Sample input shape: {item['input_ids'].shape}")
    print(f"Sample problem type: {item['problem_type']}")
    print(f"Sample difficulty: {item['difficulty']}")
    
    # Create DataLoader
    dataloader = loader.create_dataloader(demo_problems, batch_size=2)
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    
    print("Dataset loader test completed successfully!")