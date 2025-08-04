#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def analyze_precision_recall(result_file, gt_file=None):
    """Analyze precision-recall curves to find optimal thresholds."""
    
    # Load evaluation results (predictions)
    with open(result_file, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Loaded {len(predictions)} prediction frames")
    
    # Extract all scores from predictions
    all_scores = []
    for frame_pred in predictions:
        if 'score' in frame_pred and len(frame_pred['score']) > 0:
            scores = frame_pred['score']
            if hasattr(scores, 'cpu'):
                scores = scores.cpu().numpy()
            all_scores.extend(scores)
    
    all_scores = np.array(all_scores)
    print(f"Total detections: {len(all_scores)}")
    print(f"Score range: {all_scores.min():.3f} to {all_scores.max():.3f}")
    print(f"Score statistics:")
    print(f"  Mean: {all_scores.mean():.3f}")
    print(f"  Median: {np.median(all_scores):.3f}")
    print(f"  Std: {all_scores.std():.3f}")
    
    # Analyze score distribution
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    score_percentiles = np.percentile(all_scores, percentiles)
    
    print(f"\nScore Percentiles:")
    for p, score in zip(percentiles, score_percentiles):
        print(f"  {p:2d}th percentile: {score:.3f}")
    
    # Suggest thresholds based on score distribution
    suggest_conservative = np.percentile(all_scores, 80)  # Top 20%
    suggest_balanced = np.percentile(all_scores, 70)      # Top 30%  
    suggest_liberal = np.percentile(all_scores, 60)       # Top 40%
    
    print(f"\nSuggested Thresholds (based on score distribution):")
    print(f"  Conservative (top 20%): {suggest_conservative:.3f}")
    print(f"  Balanced (top 30%):     {suggest_balanced:.3f}")
    print(f"  Liberal (top 40%):      {suggest_liberal:.3f}")
    
    return predictions, all_scores

def find_optimal_threshold(precisions, recalls, thresholds):
    """Find optimal threshold where precision and recall intersect or F1 is maximized."""
    
    # Calculate F1 scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # Find threshold with maximum F1
    max_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[max_f1_idx]
    max_f1 = f1_scores[max_f1_idx]
    
    # Find intersection point (where precision ≈ recall)
    diff = np.abs(precisions - recalls)
    intersection_idx = np.argmin(diff)
    intersection_threshold = thresholds[intersection_idx]
    
    print(f"Maximum F1 Score: {max_f1:.3f} at threshold {optimal_threshold:.3f}")
    print(f"Precision-Recall intersection at threshold {intersection_threshold:.3f}")
    print(f"  Precision: {precisions[intersection_idx]:.3f}")
    print(f"  Recall: {recalls[intersection_idx]:.3f}")
    
    return optimal_threshold, intersection_threshold, max_f1

if __name__ == "__main__":
    result_file = "/media/darrell/X9 Pro/offline_pdn/LION/output/lion_models/lion_mamba_venti3d_8x_1f_1x_one_stride_128dim/default/eval/epoch_24/val/default/result.pkl"
    
    print("Analyzing evaluation results...")
    predictions, all_scores = analyze_precision_recall(result_file)