"""
Compression robustness testing (title report §3.3.3.2).

Re-encodes test videos at different H.264 CRF (Constant Rate Factor) values
and evaluates model performance at each compression level. A low CRF (e.g., 23)
means high quality; a high CRF (e.g., 40, 45) means heavy compression.

Expected finding (per title report): high-frequency visual artifacts may be
erased by compression, hurting visual-only models, while audio and temporal
modalities may offer more resilient signals.

Results are saved to:
  - results/docs/compression_robustness.csv
  - results/images/compression_robustness.png

Requires: FFmpeg installed and on PATH. Install with:
  - Windows:  choco install ffmpeg
  - Mac:      brew install ffmpeg
  - Linux:    apt-get install ffmpeg

Usage:
    python src/compression_robustness.py \
        --model_path results/weights/trained_model_multimodal_best.pth \
        --data_dir data/ff_plus/c23 \
        --crf_values 23 40 45 \
        --mode multimodal
"""

import os
import sys
import shutil
import argparse
import subprocess
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.datasets import DeepfakeDataset
from models.multimodal_model import MultimodalModel
from evaluate import evaluate_model
from train import collect_video_files


def check_ffmpeg():
    """Return True if ffmpeg is available on PATH."""
    return shutil.which('ffmpeg') is not None


def reencode_video(input_path, output_path, crf):
    """
    Re-encode a single video at a given H.264 CRF using FFmpeg.

    Args:
        input_path: Source video path.
        output_path: Destination video path.
        crf: Constant Rate Factor (0-51; lower = higher quality).
    """
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-c:v', 'libx264', '-crf', str(crf),
        '-preset', 'fast',
        '-c:a', 'copy',
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed for {input_path} (CRF={crf}): {result.stderr[:500]}"
        )


def reencode_dataset(video_paths, labels, crf, temp_dir):
    """
    Re-encode all videos at a given CRF into temp_dir, preserving the
    real/ and fake/ subdirectory structure so collect_video_files can
    find them.

    Returns the path to the compressed dataset directory.
    """
    compressed_dir = os.path.join(temp_dir, f"crf_{crf}")
    for category in ('real', 'fake'):
        os.makedirs(os.path.join(compressed_dir, category), exist_ok=True)

    for i, (src, label) in enumerate(zip(video_paths, labels)):
        category = 'real' if label == 0 else 'fake'
        dst = os.path.join(compressed_dir, category, f"video_{i:04d}.mp4")
        try:
            reencode_video(src, dst, crf)
        except RuntimeError as e:
            print(f"  Warning: {e}")
    return compressed_dir


def load_model(model_path, mode, device, **model_kwargs):
    """Load a trained model from a checkpoint."""
    if mode == 'multimodal':
        model = MultimodalModel(pretrained=False, **model_kwargs)
    else:
        from train import SimpleModel
        model = SimpleModel()

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        try:
            model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        except RuntimeError as e:
            print(f"Warning: could not load full state dict: {e}")
    else:
        print(f"Warning: model file not found at {model_path}. Using untrained weights.")
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description='Compression robustness testing (title report §3.3.3.2)'
    )
    parser.add_argument('--model_path', type=str,
                       default='results/weights/trained_model_multimodal_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Dataset directory (real/ and fake/ subdirs)')
    parser.add_argument('--crf_values', type=int, nargs='+', default=[23, 40, 45],
                       help='H.264 CRF values to test (lower=quality, higher=compression)')
    parser.add_argument('--mode', type=str, default='multimodal',
                       choices=['multimodal', 'simple', 'audio'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_csv', type=str,
                       default='results/docs/compression_robustness.csv')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of videos to re-encode (for quick testing)')
    # Model architecture flags
    parser.add_argument('--fusion_type', type=str, default='attention',
                       choices=['attention', 'concat'])
    parser.add_argument('--use_frequency', action='store_true', default=True)
    parser.add_argument('--no_frequency', dest='use_frequency', action='store_false')
    parser.add_argument('--streams', type=str, nargs='+',
                       default=['visual', 'audio', 'temporal'])
    args = parser.parse_args()

    if not check_ffmpeg():
        print("ERROR: FFmpeg is not installed or not on PATH.")
        print("Install it first:")
        print("  Windows: choco install ffmpeg")
        print("  Mac:     brew install ffmpeg")
        print("  Linux:   apt-get install ffmpeg")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model_kwargs = {}
    if args.mode == 'multimodal':
        model_kwargs = {
            'streams': tuple(args.streams),
            'fusion_type': args.fusion_type,
            'use_frequency': args.use_frequency,
        }
    model = load_model(args.model_path, args.mode, device, **model_kwargs)

    # Collect source videos
    video_files, labels = collect_video_files(args.data_dir, args.max_samples)
    if not video_files:
        print(f"No video files found in {args.data_dir}")
        return
    print(f"Source videos: {len(video_files)}")

    results = []

    # Baseline: evaluate on original (uncompressed) data
    print(f"\n--- Baseline (original quality) ---")
    dataset = DeepfakeDataset(video_paths=video_files, labels=labels, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    metrics, _, _ = evaluate_model(model, loader, device)
    metrics['crf'] = 'original'
    results.append(metrics)
    print(f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

    # Re-encode at each CRF and evaluate
    with tempfile.TemporaryDirectory() as temp_dir:
        for crf in args.crf_values:
            print(f"\n--- CRF {crf} (compression level) ---")
            try:
                compressed_dir = reencode_dataset(video_files, labels, crf, temp_dir)
                # Collect the compressed video paths
                comp_files, comp_labels = collect_video_files(compressed_dir)
                if not comp_files:
                    print(f"  No videos were successfully re-encoded at CRF={crf}")
                    continue

                comp_dataset = DeepfakeDataset(
                    video_paths=comp_files, labels=comp_labels, augment=False
                )
                comp_loader = DataLoader(comp_dataset, batch_size=args.batch_size, shuffle=False)
                metrics, _, _ = evaluate_model(model, comp_loader, device)
                metrics['crf'] = crf
                results.append(metrics)
                print(f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
            except Exception as e:
                print(f"  Failed at CRF={crf}: {e}")

    if len(results) < 2:
        print("Not enough data points for analysis.")
        return

    # Save results
    df = pd.DataFrame(results)
    cols = ['crf', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    df = df[[c for c in cols if c in df.columns]]

    docs_dir = os.path.dirname(args.output_csv)
    os.makedirs(docs_dir, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print(df.to_string(index=False))

    # Generate robustness figure
    images_dir = os.path.join(os.path.dirname(docs_dir), "images")
    os.makedirs(images_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Metrics vs CRF
    ax = axes[0]
    x_labels = [str(c) for c in df['crf']]
    x = range(len(df))
    for metric, color in [('accuracy', '#FF6B6B'), ('f1', '#4ECDC4'), ('auc', '#45B7D1')]:
        if metric in df.columns:
            ax.plot(x, df[metric], marker='o', linewidth=2, label=metric.capitalize(), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('CRF (Compression Level)')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance vs Compression')
    ax.legend()
    ax.grid(alpha=0.3)

    # Performance degradation
    ax = axes[1]
    if 'accuracy' in df.columns and len(df) > 1:
        baseline_acc = df.iloc[0]['accuracy']
        df['accuracy_drop'] = baseline_acc - df['accuracy']
        ax.bar(x_labels, df['accuracy_drop'], color='#F9A826')
        ax.set_xlabel('CRF (Compression Level)')
        ax.set_ylabel('Accuracy Drop (from baseline)')
        ax.set_title('Performance Degradation Under Compression')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(images_dir, "compression_robustness.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")

    # Robustness analysis
    if len(df) >= 2:
        print(f"\n--- Robustness Analysis ---")
        baseline_acc = df.iloc[0]['accuracy']
        worst_acc = df['accuracy'].min()
        degradation = baseline_acc - worst_acc
        print(f"Baseline accuracy:  {baseline_acc:.4f}")
        print(f"Worst accuracy:     {worst_acc:.4f}")
        print(f"Total degradation:  {degradation:.4f} ({degradation/max(baseline_acc, 1e-6)*100:.1f}%)")
        if degradation > 0.15:
            print("WARNING: Model is sensitive to compression (>15% degradation).")
            print("Consider: training on augmented/compressed data, or relying more on")
            print("audio/temporal streams that may be more robust.")


if __name__ == '__main__':
    main()
