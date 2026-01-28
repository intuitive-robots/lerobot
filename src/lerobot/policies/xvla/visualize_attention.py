#!/usr/bin/env python
"""
XVLA Attention Heatmap Visualization Script

This script loads attention weight data collected during XVLA inference
and generates heatmap visualizations.

Usage:
    python visualize_attention.py --input attention_data.pkl --output heatmaps/

    # With specific options:
    python visualize_attention.py --input attention_data.pkl \
        --output heatmaps/ \
        --samples 0,1,2 \
        --layers 0,5,10 \
        --aggregate mean \
        --format png
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def load_attention_data(filepath: str) -> dict:
    """Load attention data from pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def aggregate_heads(attention_weights: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Aggregate attention weights across heads.

    Args:
        attention_weights: Shape [batch, heads, seq_len, seq_len]
        method: Aggregation method ('mean', 'max', 'min')

    Returns:
        Aggregated attention weights [batch, seq_len, seq_len]
    """
    if method == "mean":
        return attention_weights.mean(axis=1)
    elif method == "max":
        return attention_weights.max(axis=1)
    elif method == "min":
        return attention_weights.min(axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def create_heatmap(
    attention: np.ndarray,
    title: str = "Attention Heatmap",
    output_path: str | None = None,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "viridis",
    show_colorbar: bool = True,
    token_labels: list[str] | None = None,
) -> None:
    """
    Create a heatmap visualization of attention weights.

    Args:
        attention: 2D attention matrix [seq_len, seq_len]
        title: Plot title
        output_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size (width, height)
        cmap: Colormap name
        show_colorbar: Whether to show the colorbar
        token_labels: Optional labels for tokens
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN:
        sns.heatmap(
            attention,
            ax=ax,
            cmap=cmap,
            xticklabels=token_labels if token_labels else False,
            yticklabels=token_labels if token_labels else False,
            cbar=show_colorbar,
            square=True,
        )
    else:
        im = ax.imshow(attention, cmap=cmap, aspect="auto")
        if show_colorbar:
            plt.colorbar(im, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()


def create_multi_head_heatmap(
    attention: np.ndarray,
    title: str = "Multi-Head Attention",
    output_path: str | None = None,
    max_heads: int = 16,
    figsize_per_head: tuple[float, float] = (3, 3),
    cmap: str = "viridis",
) -> None:
    """
    Create a grid of heatmaps for each attention head.

    Args:
        attention: 3D attention matrix [heads, seq_len, seq_len]
        title: Plot title
        output_path: Path to save the figure
        max_heads: Maximum number of heads to display
        figsize_per_head: Size per subplot
        cmap: Colormap name
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    num_heads = min(attention.shape[0], max_heads)
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    figsize = (cols * figsize_per_head[0], rows * figsize_per_head[1])
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if num_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(num_heads):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        im = ax.imshow(attention[idx], cmap=cmap, aspect="auto")
        ax.set_title(f"Head {idx}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(num_heads, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved multi-head heatmap to {output_path}")
    else:
        plt.show()


def create_layer_comparison(
    attention_by_layer: dict[str, np.ndarray],
    title: str = "Attention Across Layers",
    output_path: str | None = None,
    figsize_per_layer: tuple[float, float] = (4, 4),
    cmap: str = "viridis",
    aggregate: str = "mean",
) -> None:
    """
    Create a comparison of attention patterns across layers.

    Args:
        attention_by_layer: Dict mapping layer names to attention weights
        title: Plot title
        output_path: Path to save the figure
        figsize_per_layer: Size per subplot
        cmap: Colormap name
        aggregate: How to aggregate across heads
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    layer_names = sorted(attention_by_layer.keys())
    num_layers = len(layer_names)

    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols

    figsize = (cols * figsize_per_layer[0], rows * figsize_per_layer[1])
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if num_layers == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, layer_name in enumerate(layer_names):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        attn = attention_by_layer[layer_name]
        # Aggregate across batch and heads
        if attn.ndim == 4:  # [batch, heads, seq, seq]
            attn = aggregate_heads(attn[0:1], aggregate)[0]  # Take first sample
        elif attn.ndim == 3:  # [heads, seq, seq]
            attn = aggregate_heads(attn[np.newaxis], aggregate)[0]

        im = ax.imshow(attn, cmap=cmap, aspect="auto")
        ax.set_title(layer_name.replace("policy_layer_", "L"), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(num_layers, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved layer comparison to {output_path}")
    else:
        plt.show()


def create_attention_rollout(
    attention_by_layer: dict[str, np.ndarray],
    title: str = "Attention Rollout",
    output_path: str | None = None,
    aggregate: str = "mean",
) -> None:
    """
    Create attention rollout visualization (product of attention across layers).

    Args:
        attention_by_layer: Dict mapping layer names to attention weights
        title: Plot title
        output_path: Path to save the figure
        aggregate: How to aggregate across heads
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization")

    layer_names = sorted(attention_by_layer.keys())

    rollout = None
    for layer_name in layer_names:
        attn = attention_by_layer[layer_name]
        # Aggregate across batch and heads
        if attn.ndim == 4:  # [batch, heads, seq, seq]
            attn = aggregate_heads(attn[0:1], aggregate)[0]
        elif attn.ndim == 3:  # [heads, seq, seq]
            attn = aggregate_heads(attn[np.newaxis], aggregate)[0]

        # Add residual connection (identity matrix)
        attn = 0.5 * attn + 0.5 * np.eye(attn.shape[0])
        # Normalize rows
        attn = attn / attn.sum(axis=-1, keepdims=True)

        if rollout is None:
            rollout = attn
        else:
            rollout = rollout @ attn

    create_heatmap(
        rollout,
        title=title,
        output_path=output_path,
        cmap="magma",
    )


def visualize_attention_data(
    input_file: str,
    output_dir: str,
    samples: list[int] | None = None,
    layers: list[int] | None = None,
    aggregate: str = "mean",
    fmt: str = "png",
    create_rollout: bool = True,
    create_multihead: bool = True,
    create_layer_comp: bool = True,
) -> None:
    """
    Main visualization function.

    Args:
        input_file: Path to attention data pickle file
        output_dir: Directory to save visualizations
        samples: List of sample indices to visualize. None = all.
        layers: List of layer indices to visualize. None = all.
        aggregate: Head aggregation method
        fmt: Output format (png, pdf, svg)
        create_rollout: Whether to create attention rollout visualization
        create_multihead: Whether to create multi-head visualizations
        create_layer_comp: Whether to create layer comparison
    """
    # Load data
    print(f"Loading attention data from {input_file}...")
    data = load_attention_data(input_file)
    attention_samples = data["attention_data"]
    print(f"Found {len(attention_samples)} samples")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Select samples
    if samples is None:
        samples = list(range(len(attention_samples)))
    else:
        samples = [s for s in samples if s < len(attention_samples)]

    print(f"Processing {len(samples)} samples...")

    for sample_idx in samples:
        sample_data = attention_samples[sample_idx]
        sample_id = sample_data["sample_id"]
        attention_weights = sample_data["attention_weights"]

        print(f"\nProcessing sample {sample_id}...")
        print(f"  Layers: {list(attention_weights.keys())}")

        # Filter layers if specified
        if layers is not None:
            attention_weights = {
                k: v for k, v in attention_weights.items()
                if any(f"layer_{l}" in k for l in layers)
            }

        # Create per-layer heatmaps
        for layer_name, attn in attention_weights.items():
            print(f"  Processing {layer_name}, shape: {attn.shape}")

            # Aggregated heatmap
            if attn.ndim == 4:  # [batch, heads, seq, seq]
                attn_agg = aggregate_heads(attn[0:1], aggregate)[0]
            elif attn.ndim == 3:  # [heads, seq, seq]
                attn_agg = aggregate_heads(attn[np.newaxis], aggregate)[0]
            else:
                attn_agg = attn

            create_heatmap(
                attn_agg,
                title=f"Sample {sample_id} - {layer_name} ({aggregate} across heads)",
                output_path=str(output_path / f"sample_{sample_id}_{layer_name}_aggregated.{fmt}"),
            )

            # Multi-head visualization
            if create_multihead and attn.ndim >= 3:
                heads_attn = attn[0] if attn.ndim == 4 else attn
                create_multi_head_heatmap(
                    heads_attn,
                    title=f"Sample {sample_id} - {layer_name} - All Heads",
                    output_path=str(output_path / f"sample_{sample_id}_{layer_name}_multihead.{fmt}"),
                )

        # Layer comparison
        if create_layer_comp and len(attention_weights) > 1:
            create_layer_comparison(
                attention_weights,
                title=f"Sample {sample_id} - Layer Comparison",
                output_path=str(output_path / f"sample_{sample_id}_layer_comparison.{fmt}"),
                aggregate=aggregate,
            )

        # Attention rollout
        if create_rollout and len(attention_weights) > 1:
            create_attention_rollout(
                attention_weights,
                title=f"Sample {sample_id} - Attention Rollout",
                output_path=str(output_path / f"sample_{sample_id}_rollout.{fmt}"),
                aggregate=aggregate,
            )

    print(f"\nVisualization complete! Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize XVLA attention weights as heatmaps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to attention data pickle file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="attention_heatmaps",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--samples", "-s",
        type=str,
        default=None,
        help="Comma-separated sample indices to visualize (e.g., '0,1,2'). Default: all",
    )
    parser.add_argument(
        "--layers", "-l",
        type=str,
        default=None,
        help="Comma-separated layer indices to visualize (e.g., '0,5,10'). Default: all",
    )
    parser.add_argument(
        "--aggregate", "-a",
        type=str,
        choices=["mean", "max", "min"],
        default="mean",
        help="Method to aggregate attention across heads",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output image format",
    )
    parser.add_argument(
        "--no-rollout",
        action="store_true",
        help="Skip attention rollout visualization",
    )
    parser.add_argument(
        "--no-multihead",
        action="store_true",
        help="Skip multi-head visualizations",
    )
    parser.add_argument(
        "--no-layer-comp",
        action="store_true",
        help="Skip layer comparison visualization",
    )

    args = parser.parse_args()

    # Parse sample and layer lists
    samples = None
    if args.samples:
        samples = [int(s.strip()) for s in args.samples.split(",")]

    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    # Run visualization
    visualize_attention_data(
        input_file=args.input,
        output_dir=args.output,
        samples=samples,
        layers=layers,
        aggregate=args.aggregate,
        fmt=args.format,
        create_rollout=not args.no_rollout,
        create_multihead=not args.no_multihead,
        create_layer_comp=not args.no_layer_comp,
    )


if __name__ == "__main__":
    main()
