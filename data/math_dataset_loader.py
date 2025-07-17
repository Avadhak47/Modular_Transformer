"""
Mathematical Reasoning Dataset Loader

Comprehensive dataset management for mathematical reasoning tasks including:
- MATH (Mathematics Aptitude Test of Heuristics)
- GSM8K (Grade School Math 8K)
- MathQA and other mathematical reasoning datasets
- Synthetic mathematical problem generation
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

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from transformers import AutoTokenizer
import sympy
from sympy import latex, simplify, parse_expr
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class MathematicalProblem:
    """Structure for mathematical reasoning problems."""
    problem: str
    solution: str
    answer: str
    difficulty: str = "medium"
    subject: str = "general"
    source: str = "unknown"
    
    # Enhanced fields
    problem_type: str = "word_problem"
    requires_calculation: bool = True
    mathematical_concepts: List[str] = None
    step_by_step_solution: List[str] = None
    confidence_score: float = 1.0


class MathematicalDatasetLoader:
    """
    Comprehensive loader for mathematical reasoning datasets.
    
    Supports:
    - MATH dataset (12,500 problems)
    - GSM8K dataset (8,500 problems) 
    - MathQA dataset (37,000 problems)
    - MMLU Math sections
    - Synthetic problem generation
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_problems: Optional[int] = None,
        difficulty_filter: Optional[List[str]] = None,
        subject_filter: Optional[List[str]] = None
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.max_problems = max_problems
        self.difficulty_filter = difficulty_filter
        self.subject_filter = subject_filter
        
        # Dataset configurations
        self.dataset_configs = {
            "math": {
                "name": "hendrycks/competition_math",
                "splits": ["train", "test"],
                "size": 12500,
                "format": "competition"
            },
            "gsm8k": {
                "name": "gsm8k",
                "splits": ["train", "test"],
                "size": 8500,
                "format": "grade_school"
            },
            "mathqa": {
                "name": "math_qa",
                "splits": ["train", "validation", "test"],
                "size": 37000,
                "format": "multiple_choice"
            }
        }
        
        logger.info(f"Initialized MathematicalDatasetLoader with cache: {self.cache_dir}")
    
    def load_math_dataset(self, split: str = "train") -> List[MathematicalProblem]:
        """Load the MATH competition dataset."""
        cache_file = self.cache_dir / f"math_{split}.json"
        
        if cache_file.exists():
            logger.info(f"Loading cached MATH dataset from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return [MathematicalProblem(**item) for item in cached_data]
        
        try:
            logger.info(f"Loading MATH dataset split: {split}")
            dataset = load_dataset("hendrycks/competition_math", split=split)
            
            problems = []
            for item in tqdm(dataset, desc="Processing MATH dataset"):
                # Extract final answer
                answer = self._extract_answer(item["solution"])
                
                # Determine mathematical concepts
                concepts = self._extract_concepts(item["problem"])
                
                # Create step-by-step solution
                steps = self._extract_solution_steps(item["solution"])
                
                problem = MathematicalProblem(
                    problem=item["problem"],
                    solution=item["solution"],
                    answer=answer,
                    difficulty=item["level"],
                    subject=item["type"],
                    source="math_competition",
                    problem_type="competition_math",
                    requires_calculation=True,
                    mathematical_concepts=concepts,
                    step_by_step_solution=steps,
                    confidence_score=0.95  # High confidence for competition problems
                )
                
                # Apply filters
                if self._passes_filters(problem):
                    problems.append(problem)
                
                if self.max_problems and len(problems) >= self.max_problems:
                    break
            
            # Cache the processed data
            self._cache_problems(problems, cache_file)
            logger.info(f"Loaded {len(problems)} problems from MATH dataset")
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            return self._load_fallback_math_problems()
    
    def load_gsm8k_dataset(self, split: str = "train") -> List[MathematicalProblem]:
        """Load the GSM8K grade school math dataset."""
        cache_file = self.cache_dir / f"gsm8k_{split}.json"
        
        if cache_file.exists():
            logger.info(f"Loading cached GSM8K dataset from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return [MathematicalProblem(**item) for item in cached_data]
        
        try:
            logger.info(f"Loading GSM8K dataset split: {split}")
            dataset = load_dataset("gsm8k", "main", split=split)
            
            problems = []
            for item in tqdm(dataset, desc="Processing GSM8K dataset"):
                # GSM8K has answer in #### format
                answer = self._extract_gsm8k_answer(item["answer"])
                
                # Extract concepts for grade school math
                concepts = self._extract_concepts(item["question"])
                
                # Create step-by-step solution
                steps = self._extract_solution_steps(item["answer"])
                
                problem = MathematicalProblem(
                    problem=item["question"],
                    solution=item["answer"],
                    answer=answer,
                    difficulty="elementary",
                    subject="arithmetic",
                    source="gsm8k",
                    problem_type="word_problem",
                    requires_calculation=True,
                    mathematical_concepts=concepts,
                    step_by_step_solution=steps,
                    confidence_score=0.9
                )
                
                if self._passes_filters(problem):
                    problems.append(problem)
                
                if self.max_problems and len(problems) >= self.max_problems:
                    break
            
            self._cache_problems(problems, cache_file)
            logger.info(f"Loaded {len(problems)} problems from GSM8K dataset")
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            return self._load_fallback_gsm8k_problems()
    
    def load_mathqa_dataset(self, split: str = "train") -> List[MathematicalProblem]:
        """Load the MathQA dataset."""
        cache_file = self.cache_dir / f"mathqa_{split}.json"
        
        if cache_file.exists():
            logger.info(f"Loading cached MathQA dataset from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            return [MathematicalProblem(**item) for item in cached_data]
        
        try:
            logger.info(f"Loading MathQA dataset split: {split}")
            dataset = load_dataset("math_qa", split=split)
            
            problems = []
            for item in tqdm(dataset, desc="Processing MathQA dataset"):
                # MathQA is multiple choice
                answer = item["correct"]
                
                # Create solution from rationale
                solution = item.get("rationale", "No solution provided")
                
                # Extract concepts
                concepts = self._extract_concepts(item["Problem"])
                
                problem = MathematicalProblem(
                    problem=item["Problem"],
                    solution=solution,
                    answer=answer,
                    difficulty="medium",
                    subject="algebra",
                    source="mathqa",
                    problem_type="multiple_choice",
                    requires_calculation=True,
                    mathematical_concepts=concepts,
                    step_by_step_solution=solution.split('.') if solution else [],
                    confidence_score=0.8
                )
                
                if self._passes_filters(problem):
                    problems.append(problem)
                
                if self.max_problems and len(problems) >= self.max_problems:
                    break
            
            self._cache_problems(problems, cache_file)
            logger.info(f"Loaded {len(problems)} problems from MathQA dataset")
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load MathQA dataset: {e}")
            return []
    
    def load_combined_dataset(
        self, 
        datasets: List[str] = ["math", "gsm8k"], 
        split: str = "train"
    ) -> List[MathematicalProblem]:
        """Load and combine multiple mathematical datasets."""
        all_problems = []
        
        for dataset_name in datasets:
            logger.info(f"Loading {dataset_name} dataset...")
            
            if dataset_name == "math":
                problems = self.load_math_dataset(split)
            elif dataset_name == "gsm8k":
                problems = self.load_gsm8k_dataset(split)
            elif dataset_name == "mathqa":
                problems = self.load_mathqa_dataset(split)
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            all_problems.extend(problems)
            logger.info(f"Added {len(problems)} problems from {dataset_name}")
        
        # Shuffle combined dataset
        random.shuffle(all_problems)
        
        logger.info(f"Combined dataset contains {len(all_problems)} problems")
        return all_problems
    
    def generate_synthetic_problems(self, count: int = 1000) -> List[MathematicalProblem]:
        """Generate synthetic mathematical problems for data augmentation."""
        logger.info(f"Generating {count} synthetic mathematical problems...")
        
        synthetic_problems = []
        
        for i in range(count):
            problem_type = random.choice(["arithmetic", "algebra", "geometry", "word_problem"])
            
            if problem_type == "arithmetic":
                problem = self._generate_arithmetic_problem()
            elif problem_type == "algebra":
                problem = self._generate_algebra_problem()
            elif problem_type == "geometry":
                problem = self._generate_geometry_problem()
            else:
                problem = self._generate_word_problem()
            
            synthetic_problems.append(problem)
        
        logger.info(f"Generated {len(synthetic_problems)} synthetic problems")
        return synthetic_problems
    
    def _extract_answer(self, solution: str) -> str:
        """Extract the final answer from a solution."""
        # Common answer patterns
        patterns = [
            r'\\boxed\{([^}]+)\}',  # LaTeX boxed answer
            r'#### ([0-9,.\-]+)',   # GSM8K format
            r'The answer is ([0-9,.\-]+)',
            r'Answer: ([0-9,.\-]+)',
            r'= ([0-9,.\-]+)$',     # Equation result at end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution)
            if match:
                return match.group(1).replace(',', '')
        
        # Fallback: try to find last number
        numbers = re.findall(r'-?\d+\.?\d*', solution)
        return numbers[-1] if numbers else "Unknown"
    
    def _extract_gsm8k_answer(self, solution: str) -> str:
        """Extract answer from GSM8K format (#### number)."""
        match = re.search(r'#### ([0-9,.\-]+)', solution)
        return match.group(1).replace(',', '') if match else "Unknown"
    
    def _extract_concepts(self, problem: str) -> List[str]:
        """Extract mathematical concepts from problem text."""
        concepts = []
        
        concept_keywords = {
            "arithmetic": ["add", "subtract", "multiply", "divide", "sum", "difference", "product"],
            "algebra": ["equation", "variable", "solve", "polynomial", "linear", "quadratic"],
            "geometry": ["triangle", "circle", "rectangle", "area", "perimeter", "volume"],
            "probability": ["probability", "chance", "likely", "random", "odds"],
            "statistics": ["average", "mean", "median", "mode", "standard deviation"],
            "calculus": ["derivative", "integral", "limit", "function"],
            "number_theory": ["prime", "factor", "divisible", "remainder", "modulo"],
            "combinatorics": ["permutation", "combination", "counting", "arrangements"]
        }
        
        problem_lower = problem.lower()
        for concept, keywords in concept_keywords.items():
            if any(keyword in problem_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts if concepts else ["general"]
    
    def _extract_solution_steps(self, solution: str) -> List[str]:
        """Extract step-by-step solution from solution text."""
        # Split by common step indicators
        step_indicators = [
            r'Step \d+:',
            r'\d+\.',
            r'First,',
            r'Next,',
            r'Then,',
            r'Finally,',
            r'Therefore,'
        ]
        
        steps = []
        current_step = ""
        
        sentences = re.split(r'[.!?]+', solution)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if this starts a new step
            is_new_step = any(re.match(pattern, sentence, re.IGNORECASE) for pattern in step_indicators)
            
            if is_new_step and current_step:
                steps.append(current_step.strip())
                current_step = sentence
            else:
                current_step += " " + sentence if current_step else sentence
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps if steps else [solution]
    
    def _passes_filters(self, problem: MathematicalProblem) -> bool:
        """Check if problem passes specified filters."""
        if self.difficulty_filter and problem.difficulty not in self.difficulty_filter:
            return False
        
        if self.subject_filter and problem.subject not in self.subject_filter:
            return False
        
        return True
    
    def _cache_problems(self, problems: List[MathematicalProblem], cache_file: Path):
        """Cache processed problems to file."""
        try:
            data = [problem.__dict__ for problem in problems]
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cached {len(problems)} problems to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache problems: {e}")
    
    def _generate_arithmetic_problem(self) -> MathematicalProblem:
        """Generate a synthetic arithmetic problem."""
        operations = ["+", "-", "*", "/"]
        numbers = [random.randint(1, 100) for _ in range(2)]
        operation = random.choice(operations)
        
        if operation == "/":
            # Ensure clean division
            numbers[1] = random.choice([2, 3, 4, 5, 10])
            numbers[0] = numbers[1] * random.randint(1, 20)
        
        problem = f"Calculate {numbers[0]} {operation} {numbers[1]}"
        
        if operation == "+":
            answer = numbers[0] + numbers[1]
        elif operation == "-":
            answer = numbers[0] - numbers[1]
        elif operation == "*":
            answer = numbers[0] * numbers[1]
        else:  # division
            answer = numbers[0] // numbers[1]
        
        solution = f"Step 1: {numbers[0]} {operation} {numbers[1]} = {answer}"
        
        return MathematicalProblem(
            problem=problem,
            solution=solution,
            answer=str(answer),
            difficulty="easy",
            subject="arithmetic",
            source="synthetic",
            problem_type="arithmetic",
            mathematical_concepts=["arithmetic"],
            step_by_step_solution=[solution]
        )
    
    def _generate_algebra_problem(self) -> MathematicalProblem:
        """Generate a synthetic algebra problem."""
        x_value = random.randint(1, 20)
        coefficient = random.randint(2, 10)
        constant = random.randint(1, 50)
        
        result = coefficient * x_value + constant
        
        problem = f"Solve for x: {coefficient}x + {constant} = {result}"
        solution = f"Step 1: {coefficient}x = {result} - {constant} = {result - constant}\n"
        solution += f"Step 2: x = {result - constant} / {coefficient} = {x_value}"
        
        return MathematicalProblem(
            problem=problem,
            solution=solution,
            answer=str(x_value),
            difficulty="medium",
            subject="algebra",
            source="synthetic",
            problem_type="equation",
            mathematical_concepts=["algebra", "linear_equations"],
            step_by_step_solution=solution.split('\n')
        )
    
    def _generate_geometry_problem(self) -> MathematicalProblem:
        """Generate a synthetic geometry problem."""
        shape = random.choice(["rectangle", "triangle", "circle"])
        
        if shape == "rectangle":
            length = random.randint(5, 20)
            width = random.randint(3, 15)
            area = length * width
            
            problem = f"Find the area of a rectangle with length {length} and width {width}"
            solution = f"Area = length × width = {length} × {width} = {area}"
            answer = str(area)
            
        elif shape == "triangle":
            base = random.randint(4, 20)
            height = random.randint(3, 15)
            area = (base * height) // 2
            
            problem = f"Find the area of a triangle with base {base} and height {height}"
            solution = f"Area = (base × height) / 2 = ({base} × {height}) / 2 = {area}"
            answer = str(area)
            
        else:  # circle
            radius = random.randint(2, 10)
            area = int(3.14159 * radius * radius)
            
            problem = f"Find the approximate area of a circle with radius {radius} (use π ≈ 3.14)"
            solution = f"Area = π × r² = 3.14 × {radius}² = 3.14 × {radius * radius} ≈ {area}"
            answer = str(area)
        
        return MathematicalProblem(
            problem=problem,
            solution=solution,
            answer=answer,
            difficulty="medium",
            subject="geometry",
            source="synthetic",
            problem_type="geometry",
            mathematical_concepts=["geometry", "area"],
            step_by_step_solution=[solution]
        )
    
    def _generate_word_problem(self) -> MathematicalProblem:
        """Generate a synthetic word problem."""
        scenarios = [
            "shopping", "travel", "cooking", "school", "sports"
        ]
        scenario = random.choice(scenarios)
        
        if scenario == "shopping":
            items = random.randint(3, 10)
            cost_per_item = random.randint(2, 20)
            total_cost = items * cost_per_item
            
            problem = f"Sarah buys {items} books, each costing ${cost_per_item}. How much does she spend in total?"
            solution = f"Total cost = number of books × cost per book = {items} × ${cost_per_item} = ${total_cost}"
            answer = str(total_cost)
            
        else:  # Default to travel scenario
            speed = random.randint(40, 80)
            time = random.randint(2, 6)
            distance = speed * time
            
            problem = f"A car travels at {speed} mph for {time} hours. How far does it travel?"
            solution = f"Distance = speed × time = {speed} mph × {time} hours = {distance} miles"
            answer = str(distance)
        
        return MathematicalProblem(
            problem=problem,
            solution=solution,
            answer=answer,
            difficulty="easy",
            subject="word_problems",
            source="synthetic",
            problem_type="word_problem",
            mathematical_concepts=["arithmetic", "word_problems"],
            step_by_step_solution=[solution]
        )
    
    def _load_fallback_math_problems(self) -> List[MathematicalProblem]:
        """Load fallback problems if main datasets fail."""
        logger.info("Loading fallback mathematical problems...")
        
        fallback_problems = [
            MathematicalProblem(
                problem="What is 15 + 27?",
                solution="15 + 27 = 42",
                answer="42",
                difficulty="easy",
                subject="arithmetic",
                source="fallback"
            ),
            MathematicalProblem(
                problem="Solve for x: 2x + 5 = 13",
                solution="2x = 13 - 5 = 8, so x = 4",
                answer="4",
                difficulty="medium",
                subject="algebra",
                source="fallback"
            )
        ]
        
        return fallback_problems
    
    def _load_fallback_gsm8k_problems(self) -> List[MathematicalProblem]:
        """Load fallback GSM8K-style problems."""
        logger.info("Loading fallback GSM8K-style problems...")
        
        return [
            MathematicalProblem(
                problem="Tom has 5 apples. He buys 3 more apples. How many apples does he have now?",
                solution="Tom starts with 5 apples. He buys 3 more. 5 + 3 = 8. #### 8",
                answer="8",
                difficulty="elementary",
                subject="arithmetic",
                source="fallback"
            )
        ]
    
    def get_dataset_statistics(self, problems: List[MathematicalProblem]) -> Dict[str, Any]:
        """Get comprehensive statistics about the dataset."""
        stats = {
            "total_problems": len(problems),
            "sources": {},
            "difficulties": {},
            "subjects": {},
            "problem_types": {},
            "concepts": {},
            "average_confidence": 0.0
        }
        
        if not problems:
            return stats
        
        for problem in problems:
            # Count by source
            stats["sources"][problem.source] = stats["sources"].get(problem.source, 0) + 1
            
            # Count by difficulty
            stats["difficulties"][problem.difficulty] = stats["difficulties"].get(problem.difficulty, 0) + 1
            
            # Count by subject
            stats["subjects"][problem.subject] = stats["subjects"].get(problem.subject, 0) + 1
            
            # Count by problem type
            stats["problem_types"][problem.problem_type] = stats["problem_types"].get(problem.problem_type, 0) + 1
            
            # Count concepts
            for concept in problem.mathematical_concepts or []:
                stats["concepts"][concept] = stats["concepts"].get(concept, 0) + 1
        
        # Calculate average confidence
        stats["average_confidence"] = sum(p.confidence_score for p in problems) / len(problems)
        
        return stats


class MathematicalDataset(Dataset):
    """PyTorch Dataset for mathematical reasoning problems."""
    
    def __init__(
        self,
        problems: List[MathematicalProblem],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        include_solution: bool = True
    ):
        self.problems = problems
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_solution = include_solution
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        
        # Format the input
        if self.include_solution:
            text = f"Problem: {problem.problem}\n\nSolution: {problem.solution}"
        else:
            text = f"Problem: {problem.problem}\n\nSolution:"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze() if self.include_solution else None,
            "problem_metadata": {
                "difficulty": problem.difficulty,
                "subject": problem.subject,
                "concepts": problem.mathematical_concepts,
                "answer": problem.answer
            }
        }


def create_mathematical_dataloader(
    problems: List[MathematicalProblem],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 2048,
    include_solution: bool = True,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for mathematical reasoning training."""
    dataset = MathematicalDataset(
        problems=problems,
        tokenizer=tokenizer,
        max_length=max_length,
        include_solution=include_solution
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )