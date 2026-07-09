"""
Ensemble model for deepfake detection (committee Phase 3 deliverable).

Combines predictions from multiple models (e.g., MultimodalModel + MesoNet +
XceptionNet) via weighted averaging or a learned logistic-regression
meta-classifier trained on validation-set predictions.

The ensemble often outperforms any single model because individual models
make uncorrelated errors — a visual-only model may miss audio-visual sync
artifacts, while the multimodal model may miss fine-grained spatial artifacts.

Usage:
    # Equal-weight ensemble (quick):
    python src/ensemble.py --data_dir data/custom --mode eval --weights 0.5 0.3 0.2

    # Learn meta-classifier weights on validation data:
    python src/ensemble.py --data_dir data/custom --mode train_meta --model_paths \
        results/weights/trained_model_multimodal_best.pth \
        results/weights/trained_model_mesonet_best.pth

API:
    from ensemble import EnsembleModel
    ensemble = EnsembleModel(models=[mm, meso], weights=[0.6, 0.4])
    preds = ensemble.predict(video, audio)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.multimodal_model import MultimodalModel
from models.baselines import MesoNet, XceptionNet, LipSyncDetector
from data_loader.datasets import DeepfakeDataset, split_dataset
from evaluate import evaluate_model
from train import collect_video_files, train_multimodal_model, make_balanced_sampler


class EnsembleModel:
    """
    Ensemble of deepfake detection models. Combines predictions via weighted
    average (inference-time) or a learned meta-classifier.

    Args:
        models: list of PyTorch models.
        weights: optional list of weights for weighted-average fusion.
            If None, uses equal weights. Ignored if meta_classifier is set.
        meta_classifier: a fitted sklearn classifier that takes concatenated
            model predictions as input. If set, overrides weighted averaging.
    """

    def __init__(self, models, weights=None, meta_classifier=None):
        self.models = models
        self.meta_classifier = meta_classifier
        if meta_classifier is None:
            if weights is None:
                self.weights = np.ones(len(models)) / len(models)
            else:
                w = np.array(weights, dtype=float)
                self.weights = w / w.sum()
        else:
            self.weights = None

    def predict_scores(self, video, audio, device):
        """
        Get the ensemble's prediction scores for a single batch.

        Returns:
            numpy array of shape (batch,) with scores in [0, 1].
        """
        scores = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                from models.multimodal_model import MultimodalModel as MM
                from models.baselines import LipSyncDetector as LSD
                if isinstance(model, (MM, LSD)):
                    out, _ = model.process_video(video.to(device), audio.to(device))
                else:
                    out = model.process_video(video.to(device))
                    if isinstance(out, tuple):
                        out = out[0]
                scores.append(out.cpu().numpy().flatten())
        scores = np.array(scores)  # (num_models, batch)

        if self.meta_classifier is not None:
            # Meta-classifier takes concatenated predictions as features
            features = scores.T  # (batch, num_models)
            return self.meta_classifier.predict_proba(features)[:, 1]
        else:
            return (self.weights[:, None] * scores).sum(axis=0)

    def evaluate(self, data_loader, device):
        """Evaluate the ensemble on a DataLoader."""
        all_scores = []
        all_labels = []
        for batch in data_loader:
            video = batch["video"]
            audio = batch["audio"]
            labels = batch["label"].numpy()
            scores = self.predict_scores(video, audio, device)
            all_scores.extend(scores)
            all_labels.extend(labels)

        all_scores = np.array(all_scores).flatten()
        all_labels = np.array(all_labels).flatten()
        preds = (all_scores > 0.5).astype(float)

        metrics = {
            "accuracy": accuracy_score(all_labels, preds),
            "precision": precision_score(all_labels, preds, zero_division=0),
            "recall": recall_score(all_labels, preds, zero_division=0),
            "f1": f1_score(all_labels, preds, zero_division=0),
        }
        if len(np.unique(all_labels)) == 2:
            metrics["auc"] = float(roc_auc_score(all_labels, all_scores))
        else:
            metrics["auc"] = float("nan")
        return metrics, all_scores, all_labels


def train_meta_classifier(models, val_loader, device):
    """
    Train a logistic-regression meta-classifier on the validation set's
    per-model predictions. This learns optimal ensemble weights from data
    rather than guessing them.

    Returns:
        fitted sklearn LogisticRegression instance.
    """
    all_features = []
    all_labels = []
    for model in models:
        model.eval()

    for batch in val_loader:
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].numpy()

        per_model_scores = []
        for model in models:
            with torch.no_grad():
                from models.multimodal_model import MultimodalModel as MM
                from models.baselines import LipSyncDetector as LSD
                if isinstance(model, (MM, LSD)):
                    out, _ = model.process_video(video, audio)
                else:
                    out = model.process_video(video)
                    if isinstance(out, tuple):
                        out = out[0]
                per_model_scores.append(out.cpu().numpy().flatten())

        features = np.array(per_model_scores).T  # (batch, num_models)
        all_features.append(features)
        all_labels.extend(labels)

    X = np.vstack(all_features)
    y = np.array(all_labels).flatten()

    if len(np.unique(y)) < 2:
        print("Warning: only one class in validation data — cannot train meta-classifier.")
        print("Falling back to equal weights.")
        return None

    meta = LogisticRegression(max_iter=1000)
    meta.fit(X, y)
    train_acc = meta.score(X, y)
    print(f"Meta-classifier trained — validation accuracy: {train_acc:.4f}")
    print(f"  Coefficients: {meta.coef_[0]}")
    return meta


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble model evaluation (committee Phase 3 deliverable)'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Dataset directory (real/ and fake/ subdirs)')
    parser.add_argument('--mode', type=str, default='eval',
                       choices=['eval', 'train_meta'],
                       help='eval: use fixed weights; train_meta: learn meta-classifier')
    parser.add_argument('--model_paths', type=str, nargs='+',
                       default=['results/weights/trained_model_multimodal_best.pth'],
                       help='Paths to trained model checkpoints')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                       help='Ensemble weights for --mode eval (must match number of models)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=5,
                       help='Training epochs if no checkpoint exists for a model')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect data
    video_files, labels = collect_video_files(args.data_dir, args.max_samples)
    if not video_files:
        print("No video files found.")
        return

    has_both_classes = len(set(int(l) for l in labels)) == 2
    if has_both_classes and len(video_files) >= 4:
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = \
            split_dataset(video_files, labels, ratios=(0.6, 0.2, 0.2), seed=args.seed)
    else:
        train_paths, train_labels = video_files, labels
        val_paths, val_labels = [], []
        test_paths, test_labels = video_files, labels

    # Build models — MultimodalModel + MesoNet + XceptionNet as the default ensemble
    models = []
    model_names = ['MultimodalModel', 'MesoNet', 'XceptionNet']

    # MultimodalModel
    mm = MultimodalModel(pretrained=args.pretrained).to(device)
    mm_ckpt = 'results/weights/trained_model_multimodal_best.pth'
    if os.path.exists(mm_ckpt):
        ckpt = torch.load(mm_ckpt, map_location=device, weights_only=False)
        try:
            mm.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print("Loaded MultimodalModel checkpoint")
        except RuntimeError as e:
            print(f"MultimodalModel checkpoint load failed: {e}")
    models.append(mm)

    # MesoNet
    meso = MesoNet().to(device)
    models.append(meso)

    # XceptionNet
    try:
        xcept = XceptionNet(pretrained=False).to(device)
        models.append(xcept)
    except Exception as e:
        print(f"Could not create XceptionNet: {e}")
        model_names = model_names[:2]

    print(f"Ensemble: {len(models)} models ({', '.join(model_names[:len(models)])})")

    # Optionally train models that don't have checkpoints
    if args.epochs > 0 and has_both_classes:
        train_dataset = DeepfakeDataset(train_paths, train_labels, augment=True)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=make_balanced_sampler(train_labels),
        )
        loss_fn = nn.BCELoss()

        for i, (model, name) in enumerate(zip(models, model_names)):
            if isinstance(model, MultimodalModel):
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                print(f"\nTraining {name}...")
                for epoch in range(args.epochs):
                    loss = train_multimodal_model(model, train_loader, optimizer, loss_fn, device)
                    print(f"  Epoch {epoch+1}/{args.epochs} - loss: {loss:.4f}")

    # Build ensemble
    if args.mode == 'train_meta' and val_paths:
        val_dataset = DeepfakeDataset(val_paths, val_labels, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print("\nTraining meta-classifier on validation data...")
        meta = train_meta_classifier(models, val_loader, device)
        ensemble = EnsembleModel(models, meta_classifier=meta)
    else:
        if args.weights and len(args.weights) == len(models):
            ensemble = EnsembleModel(models, weights=args.weights)
            print(f"Using weights: {ensemble.weights}")
        else:
            ensemble = EnsembleModel(models)
            print(f"Using equal weights: {ensemble.weights}")

    # Evaluate on test set
    print(f"\n--- Evaluating ensemble on test set ({len(test_paths)} samples) ---")
    test_dataset = DeepfakeDataset(test_paths, test_labels, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate individual models for comparison
    print("\nIndividual model results:")
    individual_results = []
    for model, name in zip(models, model_names[:len(models)]):
        metrics, _, _ = evaluate_model(model, test_loader, device)
        print(f"  {name}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}")
        metrics['model'] = name
        individual_results.append(metrics)

    # Evaluate ensemble
    ensemble_metrics, _, _ = ensemble.evaluate(test_loader, device)
    print(f"\n  Ensemble:  acc={ensemble_metrics['accuracy']:.4f} "
          f"f1={ensemble_metrics['f1']:.4f} auc={ensemble_metrics.get('auc', float('nan')):.4f}")

    # Save results
    ensemble_metrics['model'] = 'Ensemble'
    individual_results.append(ensemble_metrics)

    docs_dir = os.path.join("results", "docs")
    os.makedirs(docs_dir, exist_ok=True)
    df = pd.DataFrame(individual_results)
    cols = ['model', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    df = df[[c for c in cols if c in df.columns]]
    csv_path = os.path.join(docs_dir, "ensemble_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # Generate comparison figure
    images_dir = os.path.join("results", "images")
    os.makedirs(images_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(df))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F9A826']
    for i, metric in enumerate(['accuracy', 'f1', 'auc']):
        if metric in df.columns:
            ax.bar([xi + i * 0.25 for xi in x], df[metric], 0.25,
                   label=metric.capitalize(), color=colors[i])
    ax.set_xticks([xi + 0.25 for xi in x])
    ax.set_xticklabels(df['model'], rotation=20, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Ensemble vs Individual Models')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(images_dir, "ensemble_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    main()
