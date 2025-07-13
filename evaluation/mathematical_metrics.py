"""
Complete implementation of all evaluation metrics for mathematical reasoning.
"""
import torch
import torch.nn.functional as F
import numpy as np
import re
import math
from typing import List, Dict, Any, Tuple, Optional
from scipy.stats import entropy
import editdistance
from transformers import AutoTokenizer
import logging

class MathematicalReasoningEvaluator:
    """Comprehensive evaluator for all mathematical reasoning metrics."""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.logger = logging.getLogger(__name__)
    
    def exact_match_accuracy(self, predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """
        Calculate exact match accuracy as specified in the proposal.
        
        Args:
            predictions: Model generated answers
            ground_truths: Correct answers
            
        Returns:
            Comprehensive accuracy metrics
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        exact_matches = 0
        numerical_matches = 0
        normalized_matches = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truths):
            # 1. Exact string match
            if pred.strip() == truth.strip():
                exact_matches += 1
            
            # 2. Numerical equivalence check
            if self._numerical_equivalence(pred, truth):
                numerical_matches += 1
            
            # 3. Normalized match (case-insensitive, whitespace-normalized)
            pred_norm = re.sub(r'\s+', ' ', pred.lower().strip())
            truth_norm = re.sub(r'\s+', ' ', truth.lower().strip())
            if pred_norm == truth_norm:
                normalized_matches += 1
        
        return {
            "exact_match_accuracy": exact_matches / total,
            "numerical_match_accuracy": numerical_matches / total,
            "normalized_match_accuracy": normalized_matches / total,
            "total_samples": total,
            "exact_matches": exact_matches,
            "numerical_matches": numerical_matches,
            "normalized_matches": normalized_matches
        }
    
    def _numerical_equivalence(self, pred: str, truth: str, tolerance: float = 1e-6) -> bool:
        """Check if two strings represent numerically equivalent values."""
        try:
            pred_num = self._extract_number(pred)
            truth_num = self._extract_number(truth)
            
            if pred_num is not None and truth_num is not None:
                return abs(pred_num - truth_num) < tolerance
            
            return False
        except:
            return False
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical value from text, handling various formats."""
        # Remove LaTeX boxed notation
        text = re.sub(r'\\boxed\{([^}]+)\}', r'\1', text)
        
        # Handle fractions
        fraction_match = re.search(r'(-?\d+)\s*/\s*(\d+)', text)
        if fraction_match:
            return float(fraction_match.group(1)) / float(fraction_match.group(2))
        
        # Handle decimals and integers
        number_match = re.search(r'-?\d+(?:\.\d+)?', text)
        if number_match:
            return float(number_match.group())
        
        return None
    
    def reasoning_step_correctness(self, reasoning_chains: List[List[str]], 
                                 problems: List[str], 
                                 solutions: List[str]) -> Dict[str, float]:
        """
        Evaluate reasoning step correctness as specified in the proposal.
        
        This evaluates whether each reasoning step is logically valid and
        contributes to solving the problem.
        """
        total_steps = 0
        correct_steps = 0
        logically_valid_steps = 0
        informative_steps = 0
        
        for chain, problem, solution in zip(reasoning_chains, problems, solutions):
            for i, step in enumerate(chain):
                total_steps += 1
                
                # Check logical validity
                if self._is_logically_valid_step(step, problem, chain[:i]):
                    logically_valid_steps += 1
                
                # Check informativeness  
                if self._is_informative_step(step, problem, solution):
                    informative_steps += 1
                
                # Overall correctness (both valid and informative)
                if (self._is_logically_valid_step(step, problem, chain[:i]) and 
                    self._is_informative_step(step, problem, solution)):
                    correct_steps += 1
        
        return {
            "reasoning_step_correctness": correct_steps / total_steps if total_steps > 0 else 0.0,
            "logical_validity": logically_valid_steps / total_steps if total_steps > 0 else 0.0,
            "informativeness": informative_steps / total_steps if total_steps > 0 else 0.0,
            "total_steps": total_steps,
            "correct_steps": correct_steps,
            "valid_steps": logically_valid_steps,
            "informative_steps": informative_steps
        }
    
    def _is_logically_valid_step(self, step: str, problem: str, previous_steps: List[str]) -> bool:
        """Determine if a reasoning step is logically valid."""
        # Basic validity checks
        if len(step.strip()) < 5:
            return False
        
        # Check for mathematical operations or logical connectors
        math_indicators = [
            '=', '+', '-', '*', '/', '^', 
            'therefore', 'thus', 'so', 'since', 'because',
            'let', 'suppose', 'assume', 'given', 'we have',
            'substituting', 'solving', 'simplifying'
        ]
        
        has_math_reasoning = any(indicator in step.lower() for indicator in math_indicators)
        
        # Check for numerical consistency
        numbers_in_step = re.findall(r'\d+(?:\.\d+)?', step)
        has_numbers = len(numbers_in_step) > 0
        
        # A valid step should have mathematical reasoning or numerical content
        return has_math_reasoning or has_numbers
    
    def _is_informative_step(self, step: str, problem: str, solution: str) -> bool:
        """Determine if a reasoning step provides useful information."""
        # Check if step contains new information not in problem
        step_words = set(re.findall(r'\w+', step.lower()))
        problem_words = set(re.findall(r'\w+', problem.lower()))
        
        # Step should introduce new concepts or perform operations
        new_information = len(step_words - problem_words) > 0
        
        # Check if step progresses toward solution
        step_numbers = set(re.findall(r'\d+(?:\.\d+)?', step))
        solution_numbers = set(re.findall(r'\d+(?:\.\d+)?', solution))
        
        numerical_progress = len(step_numbers & solution_numbers) > 0
        
        return new_information or numerical_progress
    
    def calculate_perplexity(self, model: torch.nn.Module, 
                           texts: List[str], 
                           device: torch.device = None) -> Dict[str, float]:
        """
        Calculate perplexity as specified in the proposal.
        
        Perplexity measures how well the model predicts the mathematical reasoning sequences.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.eval()
        total_log_likelihood = 0.0
        total_tokens = 0
        perplexities = []
        
        with torch.no_grad():
            for text in texts:
                try:
                    # Tokenize text
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=512
                    ).to(device)
                    
                    input_ids = inputs["input_ids"]
                    
                    # Get model predictions
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    
                    # Calculate perplexity for this sequence
                    seq_perplexity = torch.exp(loss).item()
                    perplexities.append(seq_perplexity)
                    
                    # Accumulate for overall calculation
                    seq_length = input_ids.size(1)
                    total_log_likelihood += loss.item() * seq_length
                    total_tokens += seq_length
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating perplexity for text: {e}")
                    continue
        
        overall_perplexity = math.exp(total_log_likelihood / total_tokens) if total_tokens > 0 else float('inf')
        
        return {
            "perplexity": overall_perplexity,
            "mean_perplexity": np.mean(perplexities) if perplexities else float('inf'),
            "std_perplexity": np.std(perplexities) if perplexities else 0.0,
            "min_perplexity": np.min(perplexities) if perplexities else float('inf'),
            "max_perplexity": np.max(perplexities) if perplexities else float('inf'),
            "total_tokens": total_tokens,
            "num_sequences": len(perplexities)
        }
    
    def attention_entropy(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Calculate attention entropy as specified in the proposal.
        
        Args:
            attention_weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)
            
        Returns:
            Comprehensive attention entropy metrics
        """
        if attention_weights.dim() != 4:
            raise ValueError("Attention weights must be 4D: (batch_size, n_heads, seq_len, seq_len)")
        
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Calculate entropy for each attention head and position
        all_entropies = []
        head_entropies = []
        position_entropies = []
        
        for batch_idx in range(batch_size):
            for head_idx in range(n_heads):
                head_entropy_sum = 0.0
                
                for pos in range(seq_len):
                    # Get attention distribution for this position
                    attn_dist = attention_weights[batch_idx, head_idx, pos, :]
                    
                    # Add small epsilon for numerical stability
                    attn_dist = attn_dist + 1e-8
                    attn_dist = attn_dist / attn_dist.sum()
                    
                    # Calculate entropy: H = -sum(p * log(p))
                    entropy_val = -torch.sum(attn_dist * torch.log(attn_dist)).item()
                    all_entropies.append(entropy_val)
                    head_entropy_sum += entropy_val
                
                # Average entropy for this head
                head_entropies.append(head_entropy_sum / seq_len)
        
        # Calculate position-wise entropy (averaged across heads and batches)
        for pos in range(seq_len):
            pos_entropies = []
            for batch_idx in range(batch_size):
                for head_idx in range(n_heads):
                    attn_dist = attention_weights[batch_idx, head_idx, pos, :]
                    attn_dist = attn_dist + 1e-8
                    attn_dist = attn_dist / attn_dist.sum()
                    entropy_val = -torch.sum(attn_dist * torch.log(attn_dist)).item()
                    pos_entropies.append(entropy_val)
            position_entropies.append(np.mean(pos_entropies))
        
        # Calculate maximum possible entropy (uniform distribution)
        max_entropy = math.log(seq_len)
        
        return {
            "mean_attention_entropy": np.mean(all_entropies),
            "std_attention_entropy": np.std(all_entropies),
            "head_entropy_mean": np.mean(head_entropies),
            "head_entropy_std": np.std(head_entropies),
            "position_entropy_mean": np.mean(position_entropies),
            "position_entropy_std": np.std(position_entropies),
            "normalized_entropy": np.mean(all_entropies) / max_entropy,
            "max_possible_entropy": max_entropy,
            "entropy_efficiency": 1.0 - (np.mean(all_entropies) / max_entropy),
            "total_attention_positions": len(all_entropies)
        }
    
    def comprehensive_evaluation(self, 
                               model: torch.nn.Module,
                               predictions: List[str],
                               ground_truths: List[str],
                               reasoning_chains: List[List[str]],
                               problems: List[str],
                               solutions: List[str],
                               attention_weights: Optional[torch.Tensor] = None,
                               device: torch.device = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation with all metrics specified in the proposal.
        """
        results = {}
        
        self.logger.info("Starting comprehensive mathematical reasoning evaluation")
        
        # 1. Exact Match Accuracy
        self.logger.info("Calculating exact match accuracy...")
        results["exact_match"] = self.exact_match_accuracy(predictions, ground_truths)
        
        # 2. Reasoning Step Correctness
        self.logger.info("Evaluating reasoning step correctness...")
        results["reasoning_correctness"] = self.reasoning_step_correctness(
            reasoning_chains, problems, solutions
        )
        
        # 3. Perplexity
        self.logger.info("Calculating perplexity...")
        all_texts = problems + solutions + predictions
        results["perplexity"] = self.calculate_perplexity(model, all_texts, device)
        
        # 4. Attention Entropy (if provided)
        if attention_weights is not None:
            self.logger.info("Calculating attention entropy...")
            results["attention_entropy"] = self.attention_entropy(attention_weights)
        
        # 5. Summary metrics
        results["summary"] = {
            "overall_accuracy": results["exact_match"]["exact_match_accuracy"],
            "numerical_accuracy": results["exact_match"]["numerical_match_accuracy"],
            "step_correctness": results["reasoning_correctness"]["reasoning_step_correctness"],
            "logical_validity": results["reasoning_correctness"]["logical_validity"],
            "perplexity": results["perplexity"]["perplexity"],
            "mean_attention_entropy": results.get("attention_entropy", {}).get("mean_attention_entropy", 0.0),
            "normalized_attention_entropy": results.get("attention_entropy", {}).get("normalized_entropy", 0.0)
        }
        
        self.logger.info("Comprehensive evaluation completed")
        return results

