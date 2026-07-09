import os
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
from .video_processor import VideoProcessor, extract_frames, preprocess_frames
from .audio_processor import AudioProcessor, extract_audio_features


def split_dataset(video_paths, labels, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Split video paths and labels into train / val / test sets at the video
    level (not frame level), stratified by label.

    This prevents data leakage: every frame from a given video lives in
    exactly one split (title report §3.3.1.2).

    Args:
        video_paths: list of video file paths.
        labels: list of integer labels aligned with video_paths.
        ratios: (train, val, test) fractions. Must sum to ~1.0.
        seed: reproducibility seed.

    Returns:
        Six lists: (train_paths, train_labels, val_paths, val_labels,
                    test_paths, test_labels).
    """
    if len(video_paths) != len(labels):
        raise ValueError("video_paths and labels must have the same length")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")

    rng = random.Random(seed)
    paths = list(video_paths)
    lbls = list(labels)

    # Group indices by label so each split has a proportional class mix.
    indices_by_label = {}
    for i, lbl in enumerate(lbls):
        indices_by_label.setdefault(int(lbl), []).append(i)
    for lbl in indices_by_label:
        rng.shuffle(indices_by_label[lbl])

    train_idx, val_idx, test_idx = [], [], []
    for lbl in sorted(indices_by_label.keys()):
        idxs = indices_by_label[lbl]
        n = len(idxs)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # remainder goes to test to avoid dropping samples on rounding
        n_test = n - n_train - n_val
        if n_test < 0:  # ratios over-allocated; borrow from train
            n_train += n_test
            n_test = 0
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:n_train + n_val + n_test])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    def gather(idxs):
        return [paths[i] for i in idxs], [lbls[i] for i in idxs]

    train_p, train_l = gather(train_idx)
    val_p, val_l = gather(val_idx)
    test_p, test_l = gather(test_idx)
    return train_p, train_l, val_p, val_l, test_p, test_l


def _video_to_tensor(frames, frame_count):
    if len(frames) == 0:
        return torch.zeros((frame_count, 3, 224, 224), dtype=torch.float32)

    tensor = torch.FloatTensor(frames)
    if tensor.ndim != 4:
        raise ValueError("Video frames must have shape (frames, height, width, channels)")
    return tensor.permute(0, 3, 1, 2).contiguous()


def _audio_to_tensor(audio_features):
    audio_tensor = torch.FloatTensor(audio_features)
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(0)
    return audio_tensor

class DeepfakeDataset(Dataset):
    """Base dataset class for deepfake detection"""
    def __init__(self, video_paths, labels, num_frames=30, transform=None, augment=False):
        """
        Args:
            augment: If True, apply frame-consistent visual augmentation
                (horizontal flip, ±5° rotation, brightness/contrast jitter).
                The same parameters are applied to every frame in a clip so
                temporal coherence is preserved (title report §3.3.1.2).
                Audio and frequency inputs are never augmented.
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.augment = augment
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def _augment_clip(self, video_tensor):
        """Apply one set of random params to every frame in the clip."""
        flip = random.random() < 0.5
        angle = random.uniform(-5.0, 5.0)
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        for i in range(video_tensor.shape[0]):
            frame = video_tensor[i]
            if flip:
                frame = TF.hflip(frame)
            frame = TF.rotate(frame, angle)
            frame = TF.adjust_brightness(frame, brightness)
            frame = TF.adjust_contrast(frame, contrast)
            video_tensor[i] = frame
        return video_tensor

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Process video
        frames = self.video_processor.get_frames(video_path, self.num_frames)

        # Ensure frames are not empty
        if not frames:
            return {
                "video": torch.zeros((self.num_frames, 3, 224, 224)),
                "audio": torch.zeros((1, 128, 1)), # Empty spectrogram
                "label": torch.tensor(label, dtype=torch.float)
            }

        face_frames = [self.video_processor.crop_face(frame) for frame in frames]
        video_tensor = torch.stack([self.transform(frame) for frame in face_frames])

        if self.augment:
            video_tensor = self._augment_clip(video_tensor)

        # Process audio (never augmented per title report §3.3.1.2)
        audio_waveform = self.audio_processor.extract_audio(video_path)
        mel_spectrogram = self.audio_processor.to_mel_spectrogram(audio_waveform)
        audio_tensor = torch.from_numpy(mel_spectrogram).float().unsqueeze(0)

        return {
            "video": video_tensor,
            "audio": audio_tensor,
            "label": torch.tensor(label, dtype=torch.float)
        }

class FaceForensicsDataset(Dataset):
    """Dataset class for FaceForensics++ dataset"""
    def __init__(self, root_dir, manipulation_methods=None, compression='c23', 
                 frame_count=30, transform=None, audio_transform=None):
        self.root_dir = root_dir
        self.compression = compression
        self.frame_count = frame_count
        self.transform = transform
        self.audio_transform = audio_transform
        
        if manipulation_methods is None:
            self.manipulation_methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
        else:
            self.manipulation_methods = manipulation_methods
            
        self.video_paths = []
        self.labels = []
        
        # Original videos (label 0 - real)
        original_dir = os.path.join(root_dir, f"{self.compression}", "original_sequences", "youtube")
        if os.path.exists(original_dir):
            for video_name in os.listdir(original_dir):
                if video_name.endswith('.mp4'):
                    self.video_paths.append(os.path.join(original_dir, video_name))
                    self.labels.append(0)
        
        # Manipulated videos (label 1 - fake)
        for method in self.manipulation_methods:
            manipulated_dir = os.path.join(root_dir, f"{self.compression}", "manipulated_sequences", method)
            if os.path.exists(manipulated_dir):
                for video_name in os.listdir(manipulated_dir):
                    if video_name.endswith('.mp4'):
                        self.video_paths.append(os.path.join(manipulated_dir, video_name))
                        self.labels.append(1)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path, self.frame_count)
        frames = preprocess_frames(frames)
        audio_features = extract_audio_features(video_path)
        
        if self.transform:
            frames = self.transform(frames)
        if self.audio_transform:
            audio_features = self.audio_transform(audio_features)
        
        return {
            'video': _video_to_tensor(frames, self.frame_count),
            'audio': _audio_to_tensor(audio_features),
            'label': torch.tensor(label, dtype=torch.float),
            'path': video_path
        }

class DFDCDataset(Dataset):
    """Dataset class for Deepfake Detection Challenge (DFDC) dataset"""
    def __init__(self, root_dir, split='train', frame_count=30, 
                 transform=None, audio_transform=None, subset_size=None):
        self.root_dir = root_dir
        self.split = split
        self.frame_count = frame_count
        self.transform = transform
        self.audio_transform = audio_transform
        
        if split == 'train':
            metadata_path = os.path.join(root_dir, 'metadata.json')
        elif split == 'val':
            metadata_path = os.path.join(root_dir, 'val_metadata.json')
        else:
            metadata_path = os.path.join(root_dir, 'test_metadata.json')
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            for folder in os.listdir(root_dir):
                folder_path = os.path.join(root_dir, folder)
                if os.path.isdir(folder_path):
                    json_path = os.path.join(folder_path, 'metadata.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            self.metadata.update(json.load(f))
        
        self.video_paths = []
        self.labels = []
        for video_name, video_info in self.metadata.items():
            if 'label' in video_info:
                video_path = os.path.join(self.root_dir, video_name)
                if os.path.exists(video_path):
                    self.video_paths.append(video_path)
                    self.labels.append(1 if video_info['label'] == 'FAKE' else 0)
        
        if subset_size is not None and subset_size < len(self.video_paths):
            self.video_paths = self.video_paths[:subset_size]
            self.labels = self.labels[:subset_size]
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path, self.frame_count)
        frames = preprocess_frames(frames)
        audio_features = extract_audio_features(video_path)
        
        if self.transform:
            frames = self.transform(frames)
        if self.audio_transform:
            audio_features = self.audio_transform(audio_features)
        
        return {
            'video': _video_to_tensor(frames, self.frame_count),
            'audio': _audio_to_tensor(audio_features),
            'label': torch.tensor(label, dtype=torch.float),
            'path': video_path
        }

class WildDeepfakeDataset(Dataset):
    """Dataset class for WildDeepfake dataset"""
    def __init__(self, root_dir, frame_count=30, transform=None, audio_transform=None):
        self.root_dir = root_dir
        self.frame_count = frame_count
        self.transform = transform
        self.audio_transform = audio_transform
        
        self.video_paths = []
        self.labels = []
        
        for category, label in [("real", 0), ("fake", 1)]:
            category_dir = os.path.join(root_dir, category)
            if os.path.exists(category_dir):
                for video_name in os.listdir(category_dir):
                    if video_name.endswith(('.mp4', '.avi', '.mov')):
                        self.video_paths.append(os.path.join(category_dir, video_name))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = extract_frames(video_path, self.frame_count)
        frames = preprocess_frames(frames)
        audio_features = extract_audio_features(video_path)
        
        if self.transform:
            frames = self.transform(frames)
        if self.audio_transform:
            audio_features = self.audio_transform(audio_features)
        
        return {
            'video': _video_to_tensor(frames, self.frame_count),
            'audio': _audio_to_tensor(audio_features),
            'label': torch.tensor(label, dtype=torch.float),
            'path': video_path
        }
