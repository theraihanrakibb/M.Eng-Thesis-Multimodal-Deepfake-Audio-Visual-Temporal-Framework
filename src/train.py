import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader.datasets import DeepfakeDataset, split_dataset
from models.multimodal_model import MultimodalModel
from models.baselines import MesoNet, XceptionNet, LipSyncDetector
from tqdm import tqdm
import os
import csv
import argparse
import numpy as np

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov')

class SimpleModel(nn.Module):
    """
    Simple audio-only model for quick testing and validation
    """
    def __init__(self, input_dim=128*157, num_classes=1):
        super(SimpleModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, audio):
        # Flatten the audio spectrogram
        batch_size = audio.shape[0]
        audio_flat = audio.view(batch_size, -1)
        return self.classifier(audio_flat)

def train_simple_model(model, data_loader, optimizer, loss_fn, device):
    """Training function for simple audio-only model"""
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training (Audio-only)"):
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(audio)
        loss = loss_fn(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)

def train_multimodal_model(model, data_loader, optimizer, loss_fn, device):
    """Training function for full multimodal model"""
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training (Multimodal)"):
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(video, audio)
        loss = loss_fn(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)

def train_baseline_model(model, data_loader, optimizer, loss_fn, device, model_type):
    """
    Unified training function for the three baseline models.

    Args:
        model_type: 'mesonet', 'xceptionnet', or 'lipsync'
    """
    model.train()
    total_loss = 0

    desc = f"Training ({model_type})"
    for batch in tqdm(data_loader, desc=desc):
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if model_type == 'lipsync':
            # LipSyncDetector uses both video and audio
            outputs, _ = model.process_video(video, audio)
        else:
            # MesoNet and XceptionNet use video only
            outputs = model.process_video(video)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        # Ensure output is (batch, 1) to match labels.unsqueeze(1)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)

        loss = loss_fn(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(data_loader), 1)

def validate_model(model, data_loader, loss_fn, device, mode='multimodal'):
    """Compute average validation loss without backprop."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False):
            labels = batch["label"].to(device)
            if mode in ('multimodal', 'lipsync'):
                video = batch["video"].to(device)
                audio = batch["audio"].to(device)
                if mode == 'lipsync':
                    outputs, _ = model.process_video(video, audio)
                else:
                    outputs = model(video, audio)
            elif mode in ('mesonet', 'xceptionnet'):
                video = batch["video"].to(device)
                outputs = model.process_video(video)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            else:  # simple/audio
                audio = batch["audio"].to(device)
                outputs = model(audio)
            # Ensure output is (batch, 1) to match labels.unsqueeze(1)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = loss_fn(outputs, labels.unsqueeze(1).float())
            total_loss += loss.item()
    return total_loss / max(len(data_loader), 1)

def make_balanced_sampler(labels):
    """
    Build a WeightedRandomSampler that oversamples the minority class so the
    training batches are balanced. Addresses the recall-precision imbalance
    flagged at the mid-term evaluation (models predicted 'fake' for everything).
    """
    labels = np.asarray(labels).astype(int)
    class_counts = np.bincount(labels, minlength=2).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)  # avoid div-by-zero
    per_sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(per_sample_weights),
        num_samples=len(labels),
        replacement=True,
    )

def collect_video_files(data_dir, max_samples=None):
    """
    Collect (video_paths, labels) from a directory.

    Preferred layout:
        data_dir/real/*.mp4
        data_dir/fake/*.mp4

    Falls back to a flat video folder with label 0 so smoke tests still run.
    """
    video_files = []
    labels = []

    if os.path.exists(data_dir):
        for category, label in [("real", 0), ("fake", 1)]:
            category_dir = os.path.join(data_dir, category)
            if os.path.isdir(category_dir):
                for root, _, files in os.walk(category_dir):
                    for filename in files:
                        if filename.lower().endswith(VIDEO_EXTENSIONS):
                            video_files.append(os.path.join(root, filename))
                            labels.append(label)

        if not video_files:
            video_files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]
            labels = [0] * len(video_files)

    # If no video files found, check local test data
    if not video_files:
        local_test_data = "tests/test_data"
        if os.path.exists(local_test_data):
            video_files = [
                os.path.join(local_test_data, f)
                for f in os.listdir(local_test_data)
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]
            labels = [0] * len(video_files)

    if max_samples and len(video_files) > max_samples:
        video_files = video_files[:max_samples]
        labels = labels[:max_samples]

    return video_files, labels

def create_dataset(data_dir, max_samples=None):
    """Backward-compatible wrapper: returns a single DeepfakeDataset."""
    video_files, labels = collect_video_files(data_dir, max_samples)
    if not video_files:
        print("No video files found. Please add video files to data directory or run create_test_data.py first.")
        return None
    return DeepfakeDataset(video_paths=video_files, labels=labels)

def _init_log(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "learning_rate"])

def _log_epoch(log_path, epoch, train_loss, val_loss, lr):
    val_str = f"{val_loss:.6f}" if val_loss is not None else ""
    with open(log_path, 'a', newline='') as f:
        csv.writer(f).writerow([epoch, f"{train_loss:.6f}", val_str, f"{lr:.6e}"])

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    parser.add_argument('--mode', type=str, default='multimodal',
                       choices=['multimodal', 'simple', 'audio', 'mesonet', 'xceptionnet', 'lipsync', 'all'],
                       help='Training mode: multimodal, simple/audio, mesonet, xceptionnet, lipsync, or all (trains every model)')
    parser.add_argument('--data_dir', type=str, default='tests/test_data',
                       help='Directory containing training data')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (committee recommends 20+ for real data)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for quick testing)')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained visual backbones when available')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (epochs without val loss improvement)')
    parser.add_argument('--no_val', action='store_true',
                       help='Skip validation split and early stopping (smoke test mode)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}")

    # Collect video files and labels
    video_files, labels = collect_video_files(args.data_dir, args.max_samples)
    if not video_files:
        print("No video files found.")
        return

    # Determine whether we have both classes (needed for stratified split
    # and balanced sampling).
    has_both_classes = len(set(int(l) for l in labels)) == 2
    use_val = (not args.no_val) and has_both_classes and len(video_files) >= 4

    if use_val:
        train_paths, train_labels, val_paths, val_labels, _, _ = split_dataset(
            video_files, labels, ratios=(0.7, 0.15, 0.15), seed=args.seed
        )
        print(f"Split: {len(train_paths)} train / {len(val_paths)} val")
    else:
        train_paths, train_labels = video_files, labels
        val_paths, val_labels = [], []
        if not has_both_classes:
            print("Only one class present - skipping validation and balanced sampling.")
        else:
            print("Too few samples for a validation split - training on all data.")

    # Build datasets (augmentation only on the training split)
    train_dataset = DeepfakeDataset(video_paths=train_paths, labels=train_labels, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              sampler=make_balanced_sampler(train_labels) if has_both_classes else None,
                              shuffle=(not has_both_classes))

    val_loader = None
    if val_paths:
        val_dataset = DeepfakeDataset(video_paths=val_paths, labels=val_labels, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Determine which models to train
    if args.mode == 'all':
        modes_to_train = ['multimodal', 'mesonet', 'xceptionnet', 'lipsync']
    else:
        modes_to_train = [args.mode]

    for current_mode in modes_to_train:
        if len(modes_to_train) > 1:
            print(f"\n{'='*60}")
            print(f"Training model: {current_mode}")
            print(f"{'='*60}")

        # Create model based on mode
        if current_mode == 'multimodal':
            model = MultimodalModel(pretrained=args.pretrained).to(device)
            train_func = train_multimodal_model
            print("Using MultimodalModel (video + audio)")
        elif current_mode == 'mesonet':
            model = MesoNet().to(device)
            train_func = lambda m, dl, opt, lf, dev: train_baseline_model(m, dl, opt, lf, dev, 'mesonet')
            print("Using MesoNet (visual-only baseline)")
        elif current_mode == 'xceptionnet':
            model = XceptionNet(pretrained=False).to(device)
            train_func = lambda m, dl, opt, lf, dev: train_baseline_model(m, dl, opt, lf, dev, 'xceptionnet')
            print("Using XceptionNet (visual-only baseline)")
        elif current_mode == 'lipsync':
            model = LipSyncDetector().to(device)
            train_func = lambda m, dl, opt, lf, dev: train_baseline_model(m, dl, opt, lf, dev, 'lipsync')
            print("Using LipSyncDetector (audio-visual baseline)")
        else:  # simple or audio mode
            model = SimpleModel().to(device)
            train_func = train_simple_model
            print("Using SimpleModel (audio-only)")

        # Loss, optimizer, scheduler (re-created per model)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Logging — separate log per model when training 'all'
        log_suffix = f"_{current_mode}" if len(modes_to_train) > 1 else ""
        log_path = os.path.join("results", "docs", f"training_log{log_suffix}.csv")
        _init_log(log_path)
        print(f"Training log: {log_path}")

        # Training loop with early stopping
        weights_dir = os.path.join("results", "weights")
        os.makedirs(weights_dir, exist_ok=True)
        best_ckpt_path = os.path.join(weights_dir, f"trained_model_{current_mode}_best.pth")
        best_val_loss = float('inf')
        patience_counter = 0
        last_train_loss = float('nan')

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            train_loss = train_func(model, train_loader, optimizer, loss_fn, device)
            last_train_loss = train_loss
            current_lr = scheduler.get_last_lr()[0]

            if val_loader is not None:
                val_loss = validate_model(model, val_loader, loss_fn, device, mode=current_mode)
                print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {current_lr:.2e}")
                _log_epoch(log_path, epoch + 1, train_loss, val_loss, current_lr)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'mode': current_mode,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, best_ckpt_path)
                    print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{args.patience})")
                    if patience_counter >= args.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Train loss: {train_loss:.4f} | LR: {current_lr:.2e}")
                _log_epoch(log_path, epoch + 1, train_loss, None, current_lr)

            scheduler.step()

        print(f"\nTraining completed for {current_mode}!")

        # Save final model
        model_save_path = os.path.join(weights_dir, f"trained_model_{current_mode}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'mode': current_mode,
            'epochs': args.epochs,
            'loss': last_train_loss,
        }, model_save_path)
        print(f"Final model saved to {model_save_path}")
        if os.path.exists(best_ckpt_path):
            print(f"Best checkpoint saved to {best_ckpt_path}")

        # Free GPU memory before next model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nAll training complete! ({len(modes_to_train)} model(s))")

if __name__ == '__main__':
    main()
