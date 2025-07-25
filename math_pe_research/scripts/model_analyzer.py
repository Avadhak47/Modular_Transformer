#!/usr/bin/env python3
"""
Model Analyzer for Adaptive Checkpointing

This script analyzes saved models from adaptive checkpointing,
compares their performance, and helps with model selection.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional
import pandas as pd

class ModelAnalyzer:
    """Analyze saved models from adaptive checkpointing."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_file = self.checkpoint_dir / "model_metrics.json"
        self.models_dir = self.checkpoint_dir / "adaptive_models"
        
    def load_metrics(self) -> Dict:
        """Load model metrics from file."""
        if not self.metrics_file.exists():
            print(f"❌ Metrics file not found: {self.metrics_file}")
            return {}
        
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all saved models."""
        metrics = self.load_metrics()
        best_models = metrics.get('best_models', [])
        
        if not best_models:
            print("❌ No models found in metrics file")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for model in best_models:
            data.append({
                'path': model['path'],
                'step': model['step'],
                'samples_trained': model['samples_trained'],
                'eval_loss': model['metrics'].get('eval_loss', 0),
                'eval_accuracy': model['metrics'].get('eval_accuracy', 0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('eval_loss')  # Sort by best performance first
        return df
    
    def plot_training_progress(self, save_path: Optional[Path] = None):
        """Plot training progress showing model performance over time."""
        df = self.get_model_summary()
        
        if df.empty:
            print("❌ No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Loss over samples trained
        ax1.plot(df['samples_trained'], df['eval_loss'], 'bo-', markersize=8)
        ax1.set_xlabel('Samples Trained')
        ax1.set_ylabel('Evaluation Loss')
        ax1.set_title('Model Performance: Loss vs Samples Trained')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy over samples trained
        ax2.plot(df['samples_trained'], df['eval_accuracy'], 'ro-', markersize=8)
        ax2.set_xlabel('Samples Trained')
        ax2.set_ylabel('Evaluation Accuracy')
        ax2.set_title('Model Performance: Accuracy vs Samples Trained')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        else:
            plt.show()
    
    def print_model_comparison(self):
        """Print detailed comparison of all models."""
        df = self.get_model_summary()
        
        if df.empty:
            return
        
        print("\n🏆 Model Performance Comparison")
        print("=" * 80)
        print(f"{'Rank':<4} {'Step':<8} {'Samples':<12} {'Loss':<10} {'Accuracy':<10} {'Path'}")
        print("-" * 80)
        
        for i, (_, row) in enumerate(df.iterrows(), 1):
            path = Path(row['path']).name
            print(f"{i:<4} {row['step']:<8} {row['samples_trained']:<12,} "
                  f"{row['eval_loss']:<10.4f} {row['eval_accuracy']:<10.4f} {path}")
        
        print("\n📊 Summary Statistics:")
        print(f"   Total models: {len(df)}")
        print(f"   Best loss: {df['eval_loss'].min():.4f}")
        print(f"   Best accuracy: {df['eval_accuracy'].max():.4f}")
        print(f"   Average loss: {df['eval_loss'].mean():.4f}")
        print(f"   Average accuracy: {df['eval_accuracy'].mean():.4f}")
    
    def recommend_next_model(self, strategy: str = 'random') -> Optional[str]:
        """Recommend a model for next training phase."""
        df = self.get_model_summary()
        
        if df.empty:
            return None
        
        if strategy == 'best':
            # Select the model with lowest loss
            best_model = df.iloc[0]
            print(f"🎯 Recommended (best): {best_model['path']}")
            print(f"   Loss: {best_model['eval_loss']:.4f}")
            print(f"   Accuracy: {best_model['eval_accuracy']:.4f}")
            return best_model['path']
        
        elif strategy == 'random':
            # Select random model from top 3
            top_models = df.head(3)
            selected = top_models.sample(n=1).iloc[0]
            print(f"🎲 Recommended (random from top 3): {selected['path']}")
            print(f"   Loss: {selected['eval_loss']:.4f}")
            print(f"   Accuracy: {selected['eval_accuracy']:.4f}")
            return selected['path']
        
        elif strategy == 'recent':
            # Select most recent model
            recent_model = df.iloc[-1]
            print(f"🕒 Recommended (recent): {recent_model['path']}")
            print(f"   Loss: {recent_model['eval_loss']:.4f}")
            print(f"   Accuracy: {recent_model['eval_accuracy']:.4f}")
            return recent_model['path']
        
        else:
            print(f"❌ Unknown strategy: {strategy}")
            return None
    
    def cleanup_old_models(self, keep_best: int = 5):
        """Clean up old model files, keeping only the best ones."""
        df = self.get_model_summary()
        
        if df.empty:
            print("❌ No models to clean up")
            return
        
        # Get paths of models to keep
        models_to_keep = set(df.head(keep_best)['path'])
        
        # Find all model directories
        if not self.models_dir.exists():
            print(f"❌ Models directory not found: {self.models_dir}")
            return
        
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        
        deleted_count = 0
        for model_dir in model_dirs:
            if str(model_dir) not in models_to_keep:
                try:
                    import shutil
                    shutil.rmtree(model_dir)
                    deleted_count += 1
                    print(f"🗑️  Deleted: {model_dir.name}")
                except Exception as e:
                    print(f"❌ Failed to delete {model_dir}: {e}")
        
        print(f"✅ Cleanup complete: {deleted_count} old models deleted")
        print(f"📁 Kept {len(models_to_keep)} best models")

def main():
    """Main function for model analysis."""
    parser = argparse.ArgumentParser(description="Analyze adaptive checkpointing models")
    parser.add_argument('checkpoint_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--action', choices=['summary', 'plot', 'recommend', 'cleanup'], 
                       default='summary', help='Action to perform')
    parser.add_argument('--strategy', choices=['best', 'random', 'recent'], 
                       default='random', help='Model selection strategy')
    parser.add_argument('--keep_best', type=int, default=5, help='Number of best models to keep')
    parser.add_argument('--save_plot', type=str, help='Path to save plot')
    
    args = parser.parse_args()
    
    analyzer = ModelAnalyzer(args.checkpoint_dir)
    
    if args.action == 'summary':
        analyzer.print_model_comparison()
    
    elif args.action == 'plot':
        save_path = Path(args.save_plot) if args.save_plot else None
        analyzer.plot_training_progress(save_path)
    
    elif args.action == 'recommend':
        recommended = analyzer.recommend_next_model(args.strategy)
        if recommended:
            print(f"\n✅ Recommended model: {recommended}")
        else:
            print("\n❌ No model recommended")
    
    elif args.action == 'cleanup':
        analyzer.cleanup_old_models(args.keep_best)

if __name__ == "__main__":
    main() 