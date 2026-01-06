"""Visualization utilities for plotting classification task probability distributions.

Supports comparing outputs from multiple models (e.g., De-LoRA vs Co-LoRA vs Code-LoRA).
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_classification_comparison(
    results_list: List[List[Dict[str, Any]]],
    model_names: List[str],
    output_path: str = "classification_comparison.png",
    max_samples: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300
) -> None:
    """
    Plot classification probability comparison chart for multiple models.
    
    Args:
        results_list: List of results from multiple models, each element is the return value of get_classification_probs
        model_names: List of model names (e.g., ["De-LoRA", "Co-LoRA"])
        output_path: Output image path
        max_samples: Maximum number of samples to display (defaults to all)
        figsize: Image size (width, height)
        dpi: Image resolution
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is not installed, cannot plot. Please run: pip install matplotlib")
    
    if len(results_list) != len(model_names):
        raise ValueError(f"results_list length ({len(results_list)}) does not match model_names length ({len(model_names)})")
    
    if len(results_list) == 0:
        raise ValueError("results_list cannot be empty")
    
    # Determine number of samples (take the shortest result list length)
    n_samples = min(len(r) for r in results_list)
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    
    if n_samples == 0:
        raise ValueError("No samples to display")
    
    # Set plotting style
    plt.rcParams['font.family'] = 'serif'
    
    # Create subplots: n_samples rows, 1 + len(model_names) columns
    # First column is input text, subsequent columns are predictions from each model
    n_cols = 1 + len(model_names)
    fig, axes = plt.subplots(nrows=n_samples, ncols=n_cols, figsize=figsize)
    
    # If only one row, axes is 1D array, need to reshape
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Color definitions
    baseline_bar_color = "#C0987A"  # Tan
    correct_color = "#4682B4"  # Steel blue
    highlight_edge_color = "red"
    
    # Process each sample
    for i in range(n_samples):
        # Get first model's result as reference (for extracting input text and ground truth)
        ref_result = results_list[0][i]
        input_text = ref_result.get("input_text", "")
        ground_truth = ref_result.get("ground_truth", "")
        
        # First column: text input
        ax_text = axes[i, 0]
        wrapped_text = textwrap.fill(input_text, width=20)
        ax_text.text(
            0.05, 0.5, wrapped_text,
            ha='left', va='center',
            fontsize=10.5, fontweight='bold',
            transform=ax_text.transAxes
        )
        ax_text.set_xticks([])
        ax_text.set_yticks([])
        # Set black border
        for spine in ax_text.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # Subsequent columns: predictions from each model
        for model_idx, (model_name, results) in enumerate(zip(model_names, results_list)):
            ax = axes[i, 1 + model_idx]
            result = results[i]
            
            # Get probability distribution (sorted by probability descending)
            probs = result.get("probs", {})
            if not probs:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            
            # Extract labels and scores
            labels = list(probs.keys())
            scores = list(probs.values())
            
            # Draw bar chart
            bars = ax.bar(labels, scores, color=baseline_bar_color)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11, fontweight='bold')
            ax.get_yaxis().set_visible(False)
            
            # Set black border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            
            # Highlight ground truth label
            if ground_truth:
                for j, label in enumerate(labels):
                    if label == ground_truth:
                        bars[j].set_edgecolor(highlight_edge_color)
                        bars[j].set_linewidth(2)
                        # If first model (baseline), only add red border
                        # If subsequent models (ours), add red border and blue fill
                        if model_idx > 0:
                            bars[j].set_facecolor(correct_color)
    
    # Add column titles
    axes[0, 0].set_title("Input Text (AG News)", fontsize=15, pad=10, fontweight='bold')
    for model_idx, model_name in enumerate(model_names):
        axes[0, 1 + model_idx].set_title(model_name, fontsize=15, pad=10, fontweight='bold')
    
    # Adjust spacing
    plt.subplots_adjust(wspace=0.15, hspace=0.7)
    
    # Save image
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization chart saved to: {output_path}")
    plt.close()


def plot_single_model_probs(
    results: List[Dict[str, Any]],
    model_name: str = "Model",
    output_path: str = "classification_probs.png",
    max_samples: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 300
) -> None:
    """
    Plot classification probability distribution for a single model.
    
    Args:
        results: Return value of get_classification_probs
        model_name: Model name
        output_path: Output image path
        max_samples: Maximum number of samples to display
        figsize: Image size
        dpi: Image resolution
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is not installed, cannot plot. Please run: pip install matplotlib")
    
    n_samples = len(results)
    if max_samples is not None:
        n_samples = min(n_samples, max_samples)
    
    if n_samples == 0:
        raise ValueError("No samples to display")
    
    plt.rcParams['font.family'] = 'serif'
    
    # Create subplots: n_samples rows, 2 columns (text + probability)
    fig, axes = plt.subplots(nrows=n_samples, ncols=2, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    baseline_bar_color = "#C0987A"
    correct_color = "#4682B4"
    highlight_edge_color = "red"
    
    for i in range(n_samples):
        result = results[i]
        input_text = result.get("input_text", "")
        ground_truth = result.get("ground_truth", "")
        prediction = result.get("prediction", "")
        
        # First column: text input
        ax_text = axes[i, 0]
        wrapped_text = textwrap.fill(input_text, width=20)
        ax_text.text(
            0.05, 0.5, wrapped_text,
            ha='left', va='center',
            fontsize=10.5, fontweight='bold',
            transform=ax_text.transAxes
        )
        ax_text.set_xticks([])
        ax_text.set_yticks([])
        for spine in ax_text.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1)
        
        # Second column: probability distribution
        ax = axes[i, 1]
        probs = result.get("probs", {})
        if probs:
            labels = list(probs.keys())
            scores = list(probs.values())
            bars = ax.bar(labels, scores, color=baseline_bar_color)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=11, fontweight='bold')
            ax.get_yaxis().set_visible(False)
            
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            
            # Highlight ground truth label and prediction
            for j, label in enumerate(labels):
                if label == ground_truth:
                    bars[j].set_facecolor(correct_color)
                    bars[j].set_edgecolor(highlight_edge_color)
                    bars[j].set_linewidth(2)
    
    axes[0, 0].set_title("Input Text", fontsize=15, pad=10, fontweight='bold')
    axes[0, 1].set_title(f"{model_name} Predictions", fontsize=15, pad=10, fontweight='bold')
    
    plt.subplots_adjust(wspace=0.15, hspace=0.7)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"Visualization chart saved to: {output_path}")
    plt.close()

