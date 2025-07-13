"""
Evaluation metrics for the transformer model.
"""
import torch
import math
from typing import List, Dict, Any
from collections import Counter


def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))  # Clip to prevent overflow


def calculate_bleu(predictions: List[List[str]], references: List[List[str]], 
                  max_n: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU score for generated sequences.
    Simplified implementation for demonstration.
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    scores = []
    for pred, ref in zip(predictions, references):
        score = sentence_bleu(pred, ref, max_n)
        scores.append(score)
    
    return {
        'bleu': sum(scores) / len(scores),
        'individual_scores': scores
    }


def sentence_bleu(prediction: List[str], reference: List[str], max_n: int = 4) -> float:
    """Calculate BLEU score for a single sentence."""
    if not prediction or not reference:
        return 0.0
    
    # Calculate precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = get_ngrams(prediction, n)
        ref_ngrams = get_ngrams(reference, n)
        
        if not pred_ngrams:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = 0
        for ngram in pred_ngrams:
            if ngram in ref_ngrams:
                matches += min(pred_ngrams[ngram], ref_ngrams[ngram])
        
        precision = matches / sum(pred_ngrams.values())
        precisions.append(precision)
    
    # Calculate brevity penalty
    bp = brevity_penalty(len(prediction), len(reference))
    
    # Calculate geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0
    
    geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    return bp * geometric_mean


def get_ngrams(sequence: List[str], n: int) -> Counter:
    """Get n-grams from a sequence."""
    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i:i + n])
        ngrams.append(ngram)
    return Counter(ngrams)


def brevity_penalty(pred_len: int, ref_len: int) -> float:
    """Calculate brevity penalty for BLEU score."""
    if pred_len >= ref_len:
        return 1.0
    else:
        return math.exp(1 - ref_len / pred_len)


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                      ignore_index: int = -100) -> float:
    """Calculate token-level accuracy."""
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    return correct.sum().item() / mask.sum().item()
