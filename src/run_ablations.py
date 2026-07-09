"""
Ablation study runner (title report §3.3.2.2).

Systematically evaluates MultimodalModel variants:
  1. Modality ablation — omit one stream at a time (V+A, V+T, A+T, V, A)
  2. Fusion-mechanism ablation — attention vs concat
  3. Input-representation ablation — spatial-only vs spatial+frequency (FFT)

Each variant is trained on the same data split and evaluated on the same
test set. Results are saved to:
  - results/docs/ablation_results.csv
  - results/images/ablation_comparison.png

Usage:
    python src/run_ablations.py --data_dir data/custom --epochs 10 --batch_size 4

    # Quick smoke test (no training, just verify the configs run):
    python src/run_ablations.py --data_dir tests/test_data --no_train
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.datasets import DeepfakeDataset, split_dataset
from models.multimodal_model import MultimodalModel
from evaluate import evaluate_model
from train import (
    collect_video_files,
    train_multimodal_model,
    make_balanced_sampler,
    VIDEO_EXTENSIONS,
)

# Ablation configurations — each entry is (display_name, model_kwargs).
# The full model (V+A+T, attention, spatial+frequency) is the reference.
ABLATION_CONFIGS = [
    # Reference: full model
    ('Full (V+A+T, attn, FFT)', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    # 1. Modality ablation: omit one stream
    ('V+A (no temporal)', {
        'streams': ('visual', 'audio'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    ('V+T (no audio)', {
        'streams': ('visual', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    ('A+T (no visual fusion)', {
        'streams': ('audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': True,
    }),
    # 2. Fusion-mechanism ablation
    ('Concat fusion', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'concat',
        'use_frequency': True,
    }),
    # 3. Input-representation ablation
    ('Spatial-only (no FFT)', {
        'streams': ('visual', 'audio', 'temporal'),
        'fusion_type': 'attention',
        'use_frequency': False,
    }),
]


def run_single_ablation(name, model_kwargs, train_paths, train_labels,
                        test_paths, test_labels, args, device):
    """Train and evaluate a single ablation variant."""
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"  Config: {model_kwargs}")
    print(f"{'='*60}")

    train_dataset = DeepfakeDataset(
        video_paths=train_paths, labels=train_labels, augment=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_labels) if len(set(train_labels)) == 2 else None,
        shuffle=(len(set(train_labels)) != 2),
    )
    test_dataset = DeepfakeDataset(video_paths=test_paths, labels=test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = MultimodalModel(pretrained=args.pretrained, **model_kwargs).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.no_train:
        for epoch in range(args.epochs):
            train_loss = train_multimodal_model(
                model, train_loader, optimizer, loss_fn, device
            )
            print(f"  Epoch {epoch+1}/{args.epochs} - loss: {train_loss:.4f}")
    else:
        print("  Skipping training (--no_train)")

    metrics, _, _ = evaluate_model(model, test_loader, device)
    metrics['config'] = name
    print(f"  Results: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies (title report §3.3.2.2)')
    parser.add_argument('--data_dir', type=str, default='tests/test_data',
                       help='Directory containing data (real/ and fake/ subdirs)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs per ablation variant')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbones')
    parser.add_argument('--no_train', action='store_true',
                       help='Skip training (smoke test: just evaluate untrained models)')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    video_files, labels = collect_video_files(args.data_dir, args.max_samples)
    if not video_files:
        print("No video files found.")
        return

    has_both_classes = len(set(int(l) for l in labels)) == 2
    if has_both_classes and len(video_files) >= 4:
        train_paths, train_labels, _, _, test_paths, test_labels = split_dataset(
            video_files, labels, ratios=(0.7, 0.0, 0.3), seed=args.seed
        )
    else:
        # Too few samples or single class: train and test on same data (smoke test)
        train_paths, train_labels = video_files, labels
        test_paths, test_labels = video_files, labels
        print("Warning: insufficient data for proper split — using same data for train/test")

    print(f"Train: {len(train_paths)} samples | Test: {len(test_paths)} samples")

    all_results = []
    for name, model_kwargs in ABLATION_CONFIGS:
        try:
            metrics = run_single_ablation(
                name, model_kwargs, train_paths, train_labels,
                test_paths, test_labels, args, device
            )
            all_results.append(metrics)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'config': name, 'accuracy': float('nan'),
                                'precision': float('nan'), 'recall': float('nan'),
                                'f1': float('nan'), 'auc': float('nan')})
        # Free GPU memory between configs to avoid OOM on sequential runs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save results
    docs_dir = os.path.join("results", "docs")
    images_dir = os.path.join("results", "images")
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    df = pd.DataFrame(all_results)
    cols = ['config', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    df = df[[c for c in cols if c in df.columns]]
    csv_path = os.path.join(docs_dir, "ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # Generate comparison figure using thesis_viz for consistent styling
    from thesis_viz import MODEL_COLORS, _color_for, _safe_value

    # Capitalize column names for thesis_viz
    df_thesis = df.rename(columns={
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1',
        'auc': 'AUC',
    })
    df_thesis = df_thesis.rename(columns={'config': 'Model'})
    # Add missing metric columns as NaN for thesis_viz compatibility
    for col in ('Accuracy', 'Precision', 'Recall', 'F1', 'AUC'):
        if col not in df_thesis.columns:
            df_thesis[col] = float('nan')

    # Save the canonical form for any downstream consumers
    df_thesis[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']].to_csv(
        os.path.join(docs_dir, "ablation_results.csv"), index=False
    )

    # Use thesis_viz for the standard plots
    from thesis_viz import (
        performance_comparison as tv_performance,
        precision_recall_scatter,
    )
    tv_performance(
        os.path.join(docs_dir, "ablation_results.csv"),
        os.path.join(images_dir, "ablation_comparison.png"),
    )

    # Add a dedicated relative-impact chart
    fig, ax = plt.subplots(figsize=(10, max(3, 0.6 * len(df_thesis) + 1)))
    df_rel = df_thesis.copy()
    baseline_f1 = df_rel.iloc[0]['F1']
    if baseline_f1 > 0:
        df_rel['relative'] = df_rel['F1'] / baseline_f1
        metric_label = 'Relative F1 (Full model = 1.0)'
    else:
        baseline_acc = df_rel.iloc[0]['Accuracy']
        if baseline_acc > 0:
            df_rel['relative'] = df_rel['Accuracy'] / baseline_acc
            metric_label = 'Relative Accuracy (Full model = 1.0)'
        else:
            df_rel['relative'] = 1.0
            metric_label = 'No variation (all zero)'

    df_rel = df_rel.sort_values('relative', ascending=True)
    colors = [_color_for(m) for m in df_rel['Model']]
    bars = ax.barh(df_rel['Model'], df_rel['relative'], color=colors,
                   edgecolor='white', linewidth=1)
    for bar, val in zip(bars, df_rel['relative']):
        ax.annotate(f'{val:.3f}x', xy=(val, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=10, fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Full model baseline')
    ax.set_xlabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title('Ablation Impact (Relative to Full Model)', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right')
    plt.tight_layout()
    rel_path = os.path.join(images_dir, "ablation_relative_impact.png")
    plt.savefig(rel_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved to {rel_path}")


if __name__ == '__main__':
    main()
