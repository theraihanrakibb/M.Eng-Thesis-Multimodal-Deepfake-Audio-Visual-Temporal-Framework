"""
Cross-dataset generalization evaluation (title report §3.3.3.1).

Loads a trained model and evaluates it on a DIFFERENT dataset to test
generalization — the "leave-one-dataset-out" style evaluation.

Example protocol:
  1. Train on FF++:   python src/train.py --data_dir data/ff_plus/c23 --epochs 20
  2. Test on WildDeepfake:  python src/cross_dataset_eval.py --eval_data_dir data/wilddeepfake
  3. Test on DFDC:          python src/cross_dataset_eval.py --eval_data_dir data/dfdc

Results are saved to results/docs/cross_dataset_results.csv.

Usage:
    python src/cross_dataset_eval.py \
        --model_path results/weights/trained_model_multimodal_best.pth \
        --eval_data_dir data/wilddeepfake \
        --mode multimodal

    # Evaluate on multiple datasets at once:
    python src/cross_dataset_eval.py \
        --eval_data_dir data/wilddeepfake data/dfdc \
        --mode multimodal
"""

import os
import sys
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.datasets import DeepfakeDataset
from models.multimodal_model import MultimodalModel
from evaluate import evaluate_model
from train import collect_video_files


def load_model(model_path, mode, device, **model_kwargs):
    """Load a trained model from a checkpoint."""
    if mode == 'multimodal':
        model = MultimodalModel(pretrained=False, **model_kwargs)
    else:
        from train import SimpleModel
        model = SimpleModel()

    if not os.path.exists(model_path):
        print(f"Warning: model file not found at {model_path}. Using untrained weights.")
        return model.to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    except RuntimeError as e:
        print(f"Warning: could not load full state dict: {e}")
    return model.to(device)


def evaluate_on_dataset(model, data_dir, batch_size, device):
    """Evaluate the model on a single dataset directory."""
    video_files, labels = collect_video_files(data_dir)
    if not video_files:
        print(f"  No video files found in {data_dir}")
        return None

    print(f"  Evaluating on {len(video_files)} videos from {data_dir}")
    dataset = DeepfakeDataset(video_paths=video_files, labels=labels, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics, _, _ = evaluate_model(model, loader, device)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Cross-dataset generalization evaluation (title report §3.3.3.1)'
    )
    parser.add_argument('--model_path', type=str,
                       default='results/weights/trained_model_multimodal_best.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--eval_data_dir', type=str, nargs='+', required=True,
                       help='One or more dataset directories to evaluate on')
    parser.add_argument('--mode', type=str, default='multimodal',
                       choices=['multimodal', 'simple', 'audio'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_csv', type=str,
                       default='results/docs/cross_dataset_results.csv')
    # Model architecture flags (must match the trained model)
    parser.add_argument('--fusion_type', type=str, default='attention',
                       choices=['attention', 'concat'])
    parser.add_argument('--use_frequency', action='store_true', default=True)
    parser.add_argument('--no_frequency', dest='use_frequency', action='store_false')
    parser.add_argument('--streams', type=str, nargs='+',
                       default=['visual', 'audio', 'temporal'],
                       help='Streams the trained model uses')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_kwargs = {}
    if args.mode == 'multimodal':
        model_kwargs = {
            'streams': tuple(args.streams),
            'fusion_type': args.fusion_type,
            'use_frequency': args.use_frequency,
        }

    model = load_model(args.model_path, args.mode, device, **model_kwargs)

    results = []
    for data_dir in args.eval_data_dir:
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        print(f"\n--- Evaluating on {dataset_name} ({data_dir}) ---")
        metrics = evaluate_on_dataset(model, data_dir, args.batch_size, device)
        if metrics is not None:
            metrics['dataset'] = dataset_name
            metrics['data_dir'] = data_dir
            results.append(metrics)
            print(f"  acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")

    if not results:
        print("No datasets evaluated successfully.")
        return

    df = pd.DataFrame(results)
    cols = ['dataset', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    df = df[[c for c in cols if c in df.columns]]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    print(df.to_string(index=False))

    # Generalization analysis: performance drop across datasets
    if len(df) >= 2:
        print(f"\n--- Generalization Analysis ---")
        best_acc = df['accuracy'].max()
        worst_acc = df['accuracy'].min()
        drop = best_acc - worst_acc
        print(f"Best accuracy:  {best_acc:.4f} ({df.loc[df['accuracy'].idxmax(), 'dataset']})")
        print(f"Worst accuracy: {worst_acc:.4f} ({df.loc[df['accuracy'].idxmin(), 'dataset']})")
        print(f"Accuracy drop:  {drop:.4f} ({drop/max(best_acc, 1e-6)*100:.1f}%)")
        if drop > 0.15:
            print("WARNING: Large generalization gap (>15%). The model may be overfitting")
            print("to the training dataset's specific artifacts.")


if __name__ == '__main__':
    main()
