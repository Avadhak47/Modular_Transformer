"""
Complete data pipeline for MATH and GSM8K datasets with mathematical reasoning support.
"""
import re
import json
import torch
from typing import Dict, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class MathematicalProblem:
    """Enhanced data structure for mathematical problems with reasoning chains."""
    problem: str
    solution: str
    final_answer: str
    reasoning_steps: List[str]
    problem_type: Optional[str] = None
    difficulty_level: Optional[str] = None
    dataset_source: str = "unknown"
    
class MathematicalDatasetLoader:
    """Advanced loader for MATH and GSM8K datasets with chain-of-thought processing."""
    
    def __init__(self, tokenizer_name: str = "gpt2", max_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
    def load_gsm8k_dataset(self, split: str = "train") -> List[MathematicalProblem]:
        """Load GSM8K dataset with proper preprocessing."""
        try:
            dataset = load_dataset("openai/gsm8k", "main", split=split)
            # Handle IterableDataset which doesn't support len()
            try:
                num_examples = len(dataset) if isinstance(dataset, Dataset) else "unknown number of"
            except (TypeError, AttributeError):
                num_examples = "unknown number of"
            self.logger.info(f"Successfully loaded GSM8K {split} split with {num_examples} examples")
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K dataset: {e}")
            return []
        
        problems = []
        for idx, item in enumerate(dataset):
            try:
                # Extract reasoning steps from the solution
                solution = item['answer']
                reasoning_steps = self._extract_gsm8k_reasoning_steps(solution)
                
                # Extract the final numerical answer
                final_answer = self._extract_gsm8k_final_answer(solution)
                
                problem = MathematicalProblem(
                    problem=item['question'],
                    solution=solution,
                    final_answer=final_answer,
                    reasoning_steps=reasoning_steps,
                    dataset_source="gsm8k"
                )
                problems.append(problem)
                
            except Exception as e:
                self.logger.warning(f"Error processing GSM8K item {idx}: {e}")
                continue
                
        self.logger.info(f"Successfully processed {len(problems)} GSM8K problems")
        return problems
    
    def load_math_dataset(self, split: str = "train", max_samples: Optional[int] = None) -> List[MathematicalProblem]:
        """Load MATH dataset with comprehensive preprocessing."""
        dataset = None
        sources = [
            ("Dahoas/MATH", ["train", "test"]),
            ("HuggingFaceH4/MATH-500", ["test"])  # Only test split available
        ]

        for source, valid_splits in sources:
            if split not in valid_splits:
                continue
            try:
                dataset = load_dataset(source, split=split)
                # Handle IterableDataset which doesn't support len()
                try:
                    num_examples = len(dataset) if isinstance(dataset, Dataset) else "unknown number of"
                except (TypeError, AttributeError):
                    num_examples = "unknown number of"
                self.logger.info(f"Successfully loaded MATH dataset from {source} ({split}) with {num_examples} examples")
                break
            except Exception as e:
                self.logger.warning(f"Failed to load from {source}: {e}")
                continue

        if dataset is None:
            self.logger.error("Could not load MATH dataset from any supported source. Please download or provide the dataset manually.")
            return []

        problems = []
        # Handle IterableDataset which doesn't support len()
        try:
            total_items = len(dataset) if isinstance(dataset, Dataset) else None
            if max_samples is not None and total_items is not None:
                total_items = min(total_items, max_samples)
        except (TypeError, AttributeError):
            total_items = None

        for idx, item in enumerate(dataset):
            if max_samples and idx >= max_samples:
                break
            try:
                # Extract problem components (handle both Dahoas/MATH and H4/MATH-500 fields)
                problem_text = item.get('problem', item.get('question', ''))
                solution = item.get('solution', item.get('answer', ''))
                problem_type = item.get('type', item.get('subject', 'Unknown'))
                level = str(item.get('level', 'Unknown'))

                # Process reasoning steps
                reasoning_steps = self._extract_math_reasoning_steps(solution)

                # Extract final answer from boxed notation
                final_answer = self._extract_math_final_answer(solution)

                problem = MathematicalProblem(
                    problem=problem_text,
                    solution=solution,
                    final_answer=final_answer,
                    reasoning_steps=reasoning_steps,
                    problem_type=problem_type,
                    difficulty_level=level,
                    dataset_source="math"
                )
                problems.append(problem)
            except Exception as e:
                self.logger.warning(f"Error processing MATH item {idx}: {e}")
                continue

        self.logger.info(f"Successfully processed {len(problems)} MATH problems")
        return problems
    
    def _extract_gsm8k_reasoning_steps(self, solution: str) -> List[str]:
        """Extract reasoning steps from GSM8K solution format."""
        steps = []
        
        # Split by sentences and mathematical operations
        sentences = re.split(r'[.!?]+', solution)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out empty sentences and the final answer line
            if (len(sentence) > 10 and 
                not sentence.startswith('####') and
                any(char.isalpha() for char in sentence)):
                steps.append(sentence)
        
        return steps
    
    def _extract_gsm8k_final_answer(self, solution: str) -> str:
        """Extract final numerical answer from GSM8K format (#### number)."""
        # Look for the #### pattern
        match = re.search(r'####\s*([0-9,]+(?:\.[0-9]+)?)', solution)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback: extract last number from solution
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', solution)
        return numbers[-1] if numbers else ""
    
    def _extract_math_reasoning_steps(self, solution: str) -> List[str]:
        """Extract reasoning steps from MATH dataset solutions."""
        steps = []
        
        # Remove LaTeX boxed answer first
        solution_clean = re.sub(r'\\boxed\{[^}]+\}', '', solution)
        
        # Split by mathematical reasoning indicators
        step_patterns = [
            r'Step \d+:',
            r'First,',
            r'Next,',
            r'Then,',
            r'Therefore,',
            r'Since',
            r'We have',
            r'Note that',
            r'Observe that'
        ]
        
        # Split solution into sentences
        sentences = re.split(r'[.!?]+', solution_clean)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 15 and 
                any(char.isalpha() for char in sentence) and
                ('=' in sentence or any(pattern in sentence for pattern in step_patterns))):
                steps.append(sentence)
        
        return steps[:10]  # Limit to first 10 steps for efficiency
    
    def _extract_math_final_answer(self, solution: str) -> str:
        """Extract final answer from MATH dataset boxed format."""
        # Look for \boxed{answer} pattern
        match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if match:
            answer = match.group(1).strip()
            # Clean up common LaTeX formatting
            answer = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', answer)
            return answer
        
        return ""
    
    def create_chain_of_thought_prompt(self, problem: str) -> str:
        """Create chain-of-thought prompt for mathematical reasoning."""
        return (
            f"Solve this step by step:\n\n"
            f"Problem: {problem}\n\n"
            f"Let me think through this step by step:\n\n"
            f"Solution:"
        )
    
    def prepare_training_data(self, problems: List[MathematicalProblem]) -> Dict[str, Union[torch.Tensor, List[MathematicalProblem]]]:
        """Prepare tokenized training data with chain-of-thought format."""
        inputs = []
        targets = []
        
        for problem in problems:
            # Create chain-of-thought input
            input_text = self.create_chain_of_thought_prompt(problem.problem)
            
            # Format target with reasoning steps + final answer
            target_text = problem.solution
            
            inputs.append(input_text)
            targets.append(target_text)
        
        # Tokenize inputs and targets
        input_encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=self.max_length // 2,  # Leave room for generation
            return_tensors="pt"
        )
        
        target_encodings = self.tokenizer(
            targets,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]
        }
