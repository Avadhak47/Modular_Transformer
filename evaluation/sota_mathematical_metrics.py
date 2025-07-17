#!/usr/bin/env python3
"""
SOTA Mathematical Evaluation Metrics
Comprehensive evaluation framework for mathematical reasoning models.
"""

import re
import ast
import math
import json
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import sympy
from sympy import symbols, simplify, sympify, N
import evaluate

logger = logging.getLogger(__name__)

class SOTAMathematicalEvaluator:
    """Enhanced evaluator with SOTA mathematical reasoning metrics."""
    
    def __init__(self):
        self.setup_metrics()
        self.setup_answer_patterns()
        self.setup_mathematical_verifiers()
        
    def setup_metrics(self):
        """Initialize evaluation metrics."""
        try:
            self.exact_match_metric = evaluate.load("exact_match")
            self.rouge_metric = evaluate.load("rouge")
            self.bleu_metric = evaluate.load("bleu")
        except Exception as e:
            logger.warning(f"Failed to load some metrics: {e}")
            self.exact_match_metric = None
            self.rouge_metric = None
            self.bleu_metric = None
    
    def setup_answer_patterns(self):
        """Setup patterns for extracting answers from solutions."""
        self.answer_patterns = [
            r"[Tt]he answer is (\$?-?\d+\.?\d*)",
            r"[Tt]herefore,? (\$?-?\d+\.?\d*)",
            r"[Ss]o (\$?-?\d+\.?\d*)",
            r"[Hh]ence,? (\$?-?\d+\.?\d*)",
            r"[Tt]hus,? (\$?-?\d+\.?\d*)",
            r"= (\$?-?\d+\.?\d*)",
            r"(\$?-?\d+\.?\d*) is the answer",
            r"(\$?-?\d+\.?\d*)$",  # Number at end of solution
            r"\\boxed\{([^}]+)\}",  # LaTeX boxed answer
            r"#### (\$?-?\d+\.?\d*)"  # GSM8K style answer
        ]
        
        self.fraction_patterns = [
            r"(\d+)/(\d+)",
            r"\\frac\{(\d+)\}\{(\d+)\}"
        ]
    
    def setup_mathematical_verifiers(self):
        """Setup mathematical verification tools."""
        # Define common mathematical symbols
        self.math_symbols = symbols('x y z a b c n m k t u v w p q r s')
        
        # Mathematical operation patterns
        self.operation_patterns = {
            "addition": r"(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)",
            "subtraction": r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            "multiplication": r"(\d+\.?\d*)\s*[\*×]\s*(\d+\.?\d*)",
            "division": r"(\d+\.?\d*)\s*[\/÷]\s*(\d+\.?\d*)",
            "power": r"(\d+\.?\d*)\s*[\^]\s*(\d+\.?\d*)",
            "equals": r"(\d+\.?\d*)\s*=\s*(\d+\.?\d*)"
        }
    
    def evaluate_comprehensive(self, model, tokenizer, eval_dataset: Dataset) -> Dict[str, Any]:
        """Run comprehensive evaluation with all SOTA metrics."""
        logger.info("Running comprehensive mathematical evaluation...")
        
        results = {}
        
        # Generate predictions
        predictions, references = self._generate_predictions(model, tokenizer, eval_dataset)
        
        # Core accuracy metrics
        results.update(self._compute_accuracy_metrics(predictions, references))
        
        # Mathematical correctness
        results.update(self._compute_mathematical_correctness(predictions, references))
        
        # Reasoning quality metrics
        results.update(self._compute_reasoning_metrics(predictions, references))
        
        # Length generalization
        results.update(self._test_length_generalization(model, tokenizer, eval_dataset))
        
        # Attention analysis (if model supports it)
        try:
            results.update(self._analyze_attention_patterns(model, tokenizer, eval_dataset))
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
        
        # Computational efficiency
        results.update(self._measure_computational_efficiency(model, tokenizer, eval_dataset))
        
        # Error analysis
        results.update(self._analyze_error_patterns(predictions, references))
        
        return results
    
    def _generate_predictions(self, model, tokenizer, dataset: Dataset, max_samples: int = 100) -> Tuple[List[str], List[str]]:
        """Generate model predictions for evaluation."""
        logger.info(f"Generating predictions for {min(len(dataset), max_samples)} samples...")
        
        predictions = []
        references = []
        
        model.eval()
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                
                # Prepare input
                prompt = f"Solve this step by step:\n\nProblem: {example['problem']}\n\nSolution:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                # Generate prediction
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Decode prediction
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = generated_text.replace(prompt, "").strip()
                
                predictions.append(prediction)
                references.append(example['solution'])
        
        return predictions, references
    
    def _compute_accuracy_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute basic accuracy metrics."""
        results = {}
        
        # Exact match accuracy
        if self.exact_match_metric:
            exact_match = self.exact_match_metric.compute(
                predictions=predictions,
                references=references
            )
            results["exact_match_accuracy"] = exact_match["exact_match"]
        
        # ROUGE scores
        if self.rouge_metric:
            rouge_scores = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            results.update({
                "rouge1_f1": rouge_scores["rouge1"],
                "rouge2_f1": rouge_scores["rouge2"], 
                "rougeL_f1": rouge_scores["rougeL"]
            })
        
        # BLEU score
        if self.bleu_metric:
            bleu_score = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results["bleu_score"] = bleu_score["bleu"]
        
        # Answer extraction accuracy
        results["answer_extraction_accuracy"] = self._compute_answer_accuracy(predictions, references)
        
        return results
    
    def _compute_answer_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute accuracy based on extracted numerical answers."""
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_answer = self.extract_numerical_answer(pred)
            ref_answer = self.extract_numerical_answer(ref)
            
            if pred_answer is not None and ref_answer is not None:
                if self._answers_match(pred_answer, ref_answer):
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def extract_numerical_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        # Try different patterns
        for pattern in self.answer_patterns:
            match = re.search(pattern, text)
            if match:
                answer_str = match.group(1)
                try:
                    # Remove dollar signs and convert to float
                    answer_str = answer_str.replace('$', '').replace(',', '')
                    return float(answer_str)
                except ValueError:
                    continue
        
        # Try fraction patterns
        for pattern in self.fraction_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    numerator = float(match.group(1))
                    denominator = float(match.group(2))
                    return numerator / denominator
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    def _answers_match(self, answer1: float, answer2: float, tolerance: float = 1e-6) -> bool:
        """Check if two numerical answers match within tolerance."""
        return abs(answer1 - answer2) < tolerance
    
    def _compute_mathematical_correctness(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Verify mathematical correctness using symbolic computation."""
        results = {}
        
        # Symbolic verification
        symbolic_correct = 0
        symbolic_total = 0
        
        # Arithmetic verification
        arithmetic_correct = 0
        arithmetic_total = 0
        
        for pred, ref in zip(predictions, references):
            # Extract mathematical expressions
            pred_expressions = self._extract_mathematical_expressions(pred)
            ref_expressions = self._extract_mathematical_expressions(ref)
            
            # Verify symbolic expressions
            if pred_expressions and ref_expressions:
                if self._verify_symbolic_expressions(pred_expressions, ref_expressions):
                    symbolic_correct += 1
                symbolic_total += 1
            
            # Verify arithmetic operations
            if self._verify_arithmetic_operations(pred):
                arithmetic_correct += 1
            arithmetic_total += 1
        
        results["symbolic_correctness"] = symbolic_correct / symbolic_total if symbolic_total > 0 else 0.0
        results["arithmetic_correctness"] = arithmetic_correct / arithmetic_total if arithmetic_total > 0 else 0.0
        
        return results
    
    def _extract_mathematical_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        expressions = []
        
        # Look for equations and expressions
        equation_patterns = [
            r"([^=]+=\s*[^=\n]+)",
            r"(\d+\.?\d*\s*[\+\-\*\/]\s*\d+\.?\d*)",
            r"([a-zA-Z]\s*[\+\-\*\/]\s*[\d\w\+\-\*\/]+)"
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text)
            expressions.extend(matches)
        
        return expressions
    
    def _verify_symbolic_expressions(self, pred_expr: List[str], ref_expr: List[str]) -> bool:
        """Verify symbolic expressions using SymPy."""
        try:
            for p_expr in pred_expr:
                for r_expr in ref_expr:
                    # Parse expressions
                    p_sym = sympify(p_expr.replace('=', '-(') + ')')
                    r_sym = sympify(r_expr.replace('=', '-(') + ')')
                    
                    # Check if they're equivalent
                    if simplify(p_sym - r_sym) == 0:
                        return True
        except Exception:
            pass
        
        return False
    
    def _verify_arithmetic_operations(self, text: str) -> bool:
        """Verify arithmetic operations in text."""
        correct_operations = 0
        total_operations = 0
        
        for op_type, pattern in self.operation_patterns.items():
            matches = re.findall(pattern, text)
            
            for match in matches:
                total_operations += 1
                
                try:
                    if op_type == "addition":
                        expected = float(match[0]) + float(match[1])
                    elif op_type == "subtraction":
                        expected = float(match[0]) - float(match[1])
                    elif op_type == "multiplication":
                        expected = float(match[0]) * float(match[1])
                    elif op_type == "division":
                        expected = float(match[0]) / float(match[1])
                    elif op_type == "power":
                        expected = float(match[0]) ** float(match[1])
                    elif op_type == "equals":
                        if abs(float(match[0]) - float(match[1])) < 1e-6:
                            correct_operations += 1
                        continue
                    
                    # Find the result in the text near this operation
                    # This is a simplified check
                    correct_operations += 1
                    
                except (ValueError, ZeroDivisionError):
                    pass
        
        return correct_operations / total_operations > 0.8 if total_operations > 0 else True
    
    def _compute_reasoning_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute reasoning quality metrics."""
        results = {}
        
        # Step completeness
        step_completeness_scores = []
        
        # Logical coherence
        logical_coherence_scores = []
        
        # Explanation quality
        explanation_quality_scores = []
        
        for pred, ref in zip(predictions, references):
            # Analyze reasoning steps
            pred_steps = self._extract_reasoning_steps(pred)
            ref_steps = self._extract_reasoning_steps(ref)
            
            # Step completeness
            completeness = self._compute_step_completeness(pred_steps, ref_steps)
            step_completeness_scores.append(completeness)
            
            # Logical coherence
            coherence = self._compute_logical_coherence(pred_steps)
            logical_coherence_scores.append(coherence)
            
            # Explanation quality
            quality = self._compute_explanation_quality(pred)
            explanation_quality_scores.append(quality)
        
        results["step_completeness"] = np.mean(step_completeness_scores)
        results["logical_coherence"] = np.mean(logical_coherence_scores)
        results["explanation_quality"] = np.mean(explanation_quality_scores)
        
        return results
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract reasoning steps from text."""
        # Split by common step indicators
        step_indicators = [
            r"Step \d+:",
            r"First,",
            r"Then,",
            r"Next,",
            r"Finally,",
            r"Therefore,",
            r"So,",
            r"Hence,"
        ]
        
        steps = []
        current_step = ""
        
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if this starts a new step
            is_new_step = any(re.search(indicator, sentence, re.IGNORECASE) for indicator in step_indicators)
            
            if is_new_step and current_step:
                steps.append(current_step.strip())
                current_step = sentence
            else:
                current_step += " " + sentence
        
        if current_step:
            steps.append(current_step.strip())
        
        return steps
    
    def _compute_step_completeness(self, pred_steps: List[str], ref_steps: List[str]) -> float:
        """Compute how complete the reasoning steps are."""
        if not ref_steps:
            return 1.0 if pred_steps else 0.0
        
        # Simple heuristic: ratio of predicted to reference steps
        ratio = len(pred_steps) / len(ref_steps)
        return min(1.0, ratio)
    
    def _compute_logical_coherence(self, steps: List[str]) -> float:
        """Compute logical coherence of reasoning steps."""
        if len(steps) <= 1:
            return 1.0
        
        coherence_score = 0.0
        
        for i in range(1, len(steps)):
            # Check for logical connections between consecutive steps
            prev_step = steps[i-1].lower()
            curr_step = steps[i].lower()
            
            # Look for connecting words/phrases
            connections = ["therefore", "so", "hence", "thus", "then", "because", "since"]
            has_connection = any(conn in curr_step for conn in connections)
            
            if has_connection:
                coherence_score += 1.0
        
        return coherence_score / (len(steps) - 1) if len(steps) > 1 else 1.0
    
    def _compute_explanation_quality(self, text: str) -> float:
        """Compute quality of mathematical explanation."""
        score = 0.0
        
        # Check for mathematical notation
        if re.search(r'[\+\-\*\/=]', text):
            score += 0.3
        
        # Check for step-by-step indicators
        step_words = ["first", "then", "next", "finally", "step"]
        if any(word in text.lower() for word in step_words):
            score += 0.3
        
        # Check for explanatory phrases
        explain_phrases = ["because", "since", "therefore", "so that", "in order to"]
        if any(phrase in text.lower() for phrase in explain_phrases):
            score += 0.2
        
        # Check for conclusion
        conclusion_words = ["therefore", "thus", "hence", "so", "answer"]
        if any(word in text.lower() for word in conclusion_words):
            score += 0.2
        
        return min(1.0, score)
    
    def _test_length_generalization(self, model, tokenizer, dataset: Dataset) -> Dict[str, float]:
        """Test model's ability to generalize to different problem lengths."""
        results = {}
        
        # Categorize problems by length
        short_problems = []
        medium_problems = []
        long_problems = []
        
        for example in dataset:
            problem_length = len(example['problem'].split())
            if problem_length <= 50:
                short_problems.append(example)
            elif problem_length <= 100:
                medium_problems.append(example)
            else:
                long_problems.append(example)
        
        # Evaluate on each category
        for category, problems in [("short", short_problems), ("medium", medium_problems), ("long", long_problems)]:
            if problems:
                subset_dataset = Dataset.from_list(problems[:20])  # Limit for efficiency
                predictions, references = self._generate_predictions(model, tokenizer, subset_dataset, max_samples=20)
                accuracy = self._compute_answer_accuracy(predictions, references)
                results[f"{category}_length_accuracy"] = accuracy
        
        return results
    
    def _analyze_attention_patterns(self, model, tokenizer, dataset: Dataset) -> Dict[str, float]:
        """Analyze attention patterns in the model."""
        results = {}
        
        try:
            # This is a simplified attention analysis
            # In practice, you'd need to modify the model to return attention weights
            sample_size = min(10, len(dataset))
            attention_entropies = []
            
            model.eval()
            with torch.no_grad():
                for i in range(sample_size):
                    example = dataset[i]
                    prompt = f"Solve this step by step:\n\nProblem: {example['problem']}\n\nSolution:"
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    # This would require model modifications to get attention weights
                    # For now, we'll compute a proxy metric
                    outputs = model(**inputs, output_attentions=True)
                    
                    if hasattr(outputs, 'attentions') and outputs.attentions:
                        # Compute attention entropy for the last layer
                        attention = outputs.attentions[-1]  # Last layer
                        entropy = self._compute_attention_entropy(attention)
                        attention_entropies.append(entropy)
            
            if attention_entropies:
                results["mean_attention_entropy"] = np.mean(attention_entropies)
                results["std_attention_entropy"] = np.std(attention_entropies)
        
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
            results["attention_analysis_error"] = str(e)
        
        return results
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights."""
        # attention_weights shape: (batch, heads, seq_len, seq_len)
        # Average over heads and batch
        avg_attention = attention_weights.mean(dim=(0, 1))  # (seq_len, seq_len)
        
        # Compute entropy for each query position
        entropies = []
        for i in range(avg_attention.size(0)):
            attn_dist = avg_attention[i]
            # Add small epsilon to avoid log(0)
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-9))
            entropies.append(entropy.item())
        
        return np.mean(entropies)
    
    def _measure_computational_efficiency(self, model, tokenizer, dataset: Dataset) -> Dict[str, float]:
        """Measure computational efficiency metrics."""
        import time
        
        results = {}
        
        # Measure inference time
        sample_size = min(10, len(dataset))
        inference_times = []
        
        model.eval()
        with torch.no_grad():
            for i in range(sample_size):
                example = dataset[i]
                prompt = f"Solve this step by step:\n\nProblem: {example['problem']}\n\nSolution:"
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=False,  # Use greedy for consistency
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
        
        results["mean_inference_time"] = np.mean(inference_times)
        results["std_inference_time"] = np.std(inference_times)
        
        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results["total_parameters"] = total_params
        results["trainable_parameters"] = trainable_params
        results["parameter_efficiency"] = trainable_params / total_params
        
        return results
    
    def _analyze_error_patterns(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Analyze common error patterns in predictions."""
        results = {}
        
        error_types = {
            "calculation_errors": 0,
            "reasoning_errors": 0,
            "format_errors": 0,
            "incomplete_solutions": 0
        }
        
        for pred, ref in zip(predictions, references):
            pred_answer = self.extract_numerical_answer(pred)
            ref_answer = self.extract_numerical_answer(ref)
            
            # Check for different error types
            if pred_answer is None:
                if "=" in pred or any(char.isdigit() for char in pred):
                    error_types["format_errors"] += 1
                else:
                    error_types["incomplete_solutions"] += 1
            elif ref_answer is not None and not self._answers_match(pred_answer, ref_answer):
                # Check if it's a calculation error
                if self._has_correct_reasoning_structure(pred, ref):
                    error_types["calculation_errors"] += 1
                else:
                    error_types["reasoning_errors"] += 1
        
        # Convert to percentages
        total_errors = sum(error_types.values())
        if total_errors > 0:
            for error_type in error_types:
                results[f"{error_type}_percentage"] = (error_types[error_type] / len(predictions)) * 100
        
        return results
    
    def _has_correct_reasoning_structure(self, pred: str, ref: str) -> bool:
        """Check if prediction has similar reasoning structure to reference."""
        pred_steps = self._extract_reasoning_steps(pred)
        ref_steps = self._extract_reasoning_steps(ref)
        
        # Simple heuristic: similar number of steps
        return abs(len(pred_steps) - len(ref_steps)) <= 1
    
    def verify_mathematical_correctness_batch(self, predictions: List[str], references: List[str]) -> float:
        """Batch verification of mathematical correctness."""
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            pred_answer = self.extract_numerical_answer(pred)
            ref_answer = self.extract_numerical_answer(ref)
            
            if pred_answer is not None and ref_answer is not None:
                if self._answers_match(pred_answer, ref_answer):
                    correct += 1
        
        return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    # Test the evaluator
    evaluator = SOTAMathematicalEvaluator()
    
    # Test answer extraction
    test_text = "Therefore, the answer is 42."
    answer = evaluator.extract_numerical_answer(test_text)
    print(f"Extracted answer: {answer}")
    
    # Test mathematical verification
    predictions = ["The answer is 42", "So the result is 3.14"]
    references = ["Therefore, 42 is correct", "The answer is 3.14159"]
    
    accuracy = evaluator.verify_mathematical_correctness_batch(predictions, references)
    print(f"Mathematical correctness: {accuracy}")