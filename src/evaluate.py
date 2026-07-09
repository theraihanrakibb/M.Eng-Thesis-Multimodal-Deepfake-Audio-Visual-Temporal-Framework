import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Import models
from models.multimodal_model import MultimodalModel
from models.baselines import MesoNet, XceptionNet, LipSyncDetector

# Import advanced visualizations
from visualizations import create_visualizations

# Import media processor
from media_processor import MediaProcessor
from data_loader.audio_processor import extract_audio_features

class DummyDataset(Dataset):
    """
    Dummy dataset for testing the evaluation framework
    """
    def __init__(self, num_samples=100, num_frames=10):
        self.num_samples = num_samples
        self.num_frames = num_frames
        # Generate random labels (0 for real, 1 for fake)
        self.labels = torch.randint(0, 2, (num_samples,)).float()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random video frames (batch_size, num_frames, channels, height, width)
        video = torch.rand(self.num_frames, 3, 256, 256)
        # Generate random audio features - shape adjusted to match model expectations
        audio = torch.rand(self.num_frames, 128)
        # Get label
        label = self.labels[idx]
        
        return {
            "video": video,
            "audio": audio,
            "label": label
        }

def evaluate_model(model, data_loader, device):
    """
    Evaluate a model on a dataset
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run the model on
        
    Returns:
        tuple: (metrics, predictions, labels)
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get data
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            if isinstance(model, MultimodalModel) or isinstance(model, LipSyncDetector):
                outputs, _ = model.process_video(video, audio)
            else:
                outputs = model.process_video(video)
                
            # Handle tuple outputs
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Convert outputs to binary predictions
            preds = (outputs > 0.5).float()
            
            # Store predictions and labels
            all_scores.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_scores = np.array(all_scores).flatten()
    all_labels = np.array(all_labels).flatten()
    all_preds = (all_scores > 0.5).astype(np.float32)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    # AUC-ROC requires both classes to be present in the labels.
    if len(np.unique(all_labels)) == 2:
        metrics["auc"] = float(roc_auc_score(all_labels, all_scores))
    else:
        metrics["auc"] = float("nan")

    return metrics, all_scores, all_labels

def evaluate_on_media_files(model, media_processor, device):
    """
    Evaluate model on real media files
    
    Args:
        model: PyTorch model
        media_processor: MediaProcessor instance
        device: Device to run the model on
        
    Returns:
        Dictionary with evaluation metrics on media files
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Get media files
    media_files = media_processor.get_media_files()
    
    # Process real videos
    for video_path in media_files["real_videos"]:
        video_frames = media_processor.process_video_file(video_path)
        if video_frames is not None:
            # Add batch dimension
            video_frames = video_frames.unsqueeze(0)
            
            audio_features = torch.from_numpy(extract_audio_features(str(video_path))).float().unsqueeze(0)
            
            with torch.no_grad():
                if isinstance(model, MultimodalModel) or isinstance(model, LipSyncDetector):
                    outputs, _ = model.process_video(video_frames.to(device), audio_features.to(device))
                else:
                    outputs = model.process_video(video_frames.to(device))
                
                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert to binary prediction
                pred = (outputs > 0.5).float()
                
                # Real videos should be classified as 0
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend([0] * len(pred))
    
    # Process fake videos
    for video_path in media_files["fake_videos"]:
        video_frames = media_processor.process_video_file(video_path)
        if video_frames is not None:
            # Add batch dimension
            video_frames = video_frames.unsqueeze(0)
            
            audio_features = torch.from_numpy(extract_audio_features(str(video_path))).float().unsqueeze(0)
            
            with torch.no_grad():
                if isinstance(model, MultimodalModel) or isinstance(model, LipSyncDetector):
                    outputs, _ = model.process_video(video_frames.to(device), audio_features.to(device))
                else:
                    outputs = model.process_video(video_frames.to(device))
                
                # Handle tuple outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Convert to binary prediction
                pred = (outputs > 0.5).float()
                
                # Fake videos should be classified as 1
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend([1] * len(pred))
    
    if len(all_preds) == 0:
        print("No media files processed - using dummy results")
        return {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5
        }
    
    # Calculate metrics
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    
    return metrics

def create_media_comparisons(media_processor):
    """
    Create comparison GIFs between real and fake media
    """
    media_files = media_processor.get_media_files()
    
    # Create comparison GIFs if we have both real and fake videos
    if len(media_files["real_videos"]) > 0 and len(media_files["fake_videos"]) > 0:
        for i, (real_video, fake_video) in enumerate(zip(media_files["real_videos"][:3], media_files["fake_videos"][:3])):
            output_path = media_processor.gifs_dir / f"comparison_{i}.gif"
            media_processor.create_comparison_gif(real_video, fake_video, output_path)
    else:
        print("Not enough media files to create comparisons")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection models')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Directory with real/ and fake/ video subdirs. '
                            'If provided, evaluates on real videos instead of random DummyDataset.')
    parser.add_argument('--batch_size', type=int, default=4)
    args, _ = parser.parse_known_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize media processor
    media_processor = MediaProcessor()

    # Build test dataset: use real videos when --data_dir is provided,
    # otherwise fall back to DummyDataset (random noise — smoke test only).
    if args.data_dir and os.path.isdir(args.data_dir):
        from train import collect_video_files
        from data_loader.datasets import DeepfakeDataset
        video_files, labels = collect_video_files(args.data_dir)
        if video_files:
            print(f"Evaluating on {len(video_files)} real videos from {args.data_dir}")
            test_dataset = DeepfakeDataset(
                video_paths=video_files, labels=labels, augment=False
            )
        else:
            print(f"No videos found in {args.data_dir}, falling back to DummyDataset")
            test_dataset = DummyDataset(num_samples=50, num_frames=10)
    else:
        print("No --data_dir provided. Using DummyDataset (random noise — smoke test only).")
        print("For real results: python src/evaluate.py --data_dir data/custom")
        test_dataset = DummyDataset(num_samples=50, num_frames=10)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    def _load_trained(model, name, device):
        """Load trained checkpoint for a baseline model if it exists."""
        ckpt_path = f"results/weights/trained_model_{name}.pth"
        if os.path.exists(ckpt_path):
            print(f"Loading trained {name} from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            try:
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict)
                print(f"  {name} loaded successfully!")
            except RuntimeError as e:
                print(f"  Warning: {name} checkpoint incompatible, using random weights: {e}")
        else:
            print(f"  No trained checkpoint for {name}, using initialized weights.")
        return model

    # Initialize and load all models
    multimodal_model = _load_trained(MultimodalModel(pretrained=False).to(device), 'multimodal', device)
    mesonet_model = _load_trained(MesoNet().to(device), 'mesonet', device)
    xception_model = _load_trained(XceptionNet(pretrained=False).to(device), 'xceptionnet', device)
    lipsync_model = _load_trained(LipSyncDetector().to(device), 'lipsync', device)
    
    # Dictionary to store results
    results = {}
    predictions_dict = {}
    labels_dict = {}
    
    print("Evaluating MultimodalModel...")
    metrics, preds, labels = evaluate_model(multimodal_model, test_loader, device)
    results["MultimodalModel"] = metrics
    predictions_dict["MultimodalModel"] = preds
    labels_dict["MultimodalModel"] = labels
    
    print("Evaluating MesoNet...")
    metrics, preds, labels = evaluate_model(mesonet_model, test_loader, device)
    results["MesoNet"] = metrics
    predictions_dict["MesoNet"] = preds
    labels_dict["MesoNet"] = labels
    
    print("Evaluating XceptionNet...")
    metrics, preds, labels = evaluate_model(xception_model, test_loader, device)
    results["XceptionNet"] = metrics
    predictions_dict["XceptionNet"] = preds
    labels_dict["XceptionNet"] = labels
    
    print("Evaluating LipSyncDetector...")
    metrics, preds, labels = evaluate_model(lipsync_model, test_loader, device)
    results["LipSyncDetector"] = metrics
    predictions_dict["LipSyncDetector"] = preds
    labels_dict["LipSyncDetector"] = labels
    
    # Evaluate on media files
    print("Evaluating media files...")
    media_results = {}
    
    print("Evaluating MesoNet on media files...")
    media_results["MesoNet"] = evaluate_on_media_files(mesonet_model, media_processor, device)
    
    print("Evaluating XceptionNet on media files...")
    media_results["XceptionNet"] = evaluate_on_media_files(xception_model, media_processor, device)
    
    print("Evaluating LipSyncDetector on media files...")
    media_results["LipSyncDetector"] = evaluate_on_media_files(lipsync_model, media_processor, device)
    
    # Print media results
    print("\nMedia File Evaluation Results:")
    for model_name, metrics in media_results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Create media comparisons (GIFs)
    print("\nCreating media comparisons...")
    create_media_comparisons(media_processor)
    
    # Print results
    print("Synthetic data evaluation results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(results, predictions_dict, labels_dict)
    print("Visualizations saved to 'results' directory")

    # Save raw predictions + labels so main.py --thesis can render
    # ROC curves and confusion matrices from real scores.
    import json
    raw_path = os.path.join("results", "docs", "raw_predictions.json")
    serializable_preds = {
        m: np.asarray(v).flatten().tolist() for m, v in predictions_dict.items()
    }
    serializable_labels = {
        m: np.asarray(v).flatten().astype(int).tolist() for m, v in labels_dict.items()
    }
    with open(raw_path, 'w') as f:
        json.dump({
            'predictions': serializable_preds,
            'labels': serializable_labels,
        }, f)
    print(f"Raw predictions saved to {raw_path}")
    
    # Show media files information
    media_files = media_processor.get_media_files()
    print("\n===== MEDIA FILES INFORMATION =====")
    for category, files in media_files.items():
        print(f"{category}: {len(files)} files")
        for file in files[:3]:  # Show first 3 files of each category
            print(f"  - {file.name}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more")
    
    print("\nMedia comparisons saved to 'media/gifs/' directory")

if __name__ == "__main__":
    main()
