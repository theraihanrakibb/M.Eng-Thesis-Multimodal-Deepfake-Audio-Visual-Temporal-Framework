import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pretrainedmodels

class MesoNet(nn.Module):
    """
    Implementation of MesoNet model for deepfake detection
    MesoNet is a CNN architecture specifically designed to detect facial manipulations
    at a mesoscopic level of analysis.
    """
    def __init__(self):
        super(MesoNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def process_video(self, frames_batch):
        batch_size, num_frames = frames_batch.shape[:2]
        frames_reshaped = frames_batch.view(-1, *frames_batch.shape[2:])
        if frames_reshaped.shape[2] != 256 or frames_reshaped.shape[3] != 256:
            frames_reshaped = F.interpolate(frames_reshaped, size=(256, 256), mode='bilinear', align_corners=False)
        frame_preds = self.forward(frames_reshaped)
        frame_preds = frame_preds.view(batch_size, num_frames)
        return torch.mean(frame_preds, dim=1)

class XceptionNet(nn.Module):
    """
    Implementation of XceptionNet model for deepfake detection
    XceptionNet is commonly used as a strong visual baseline on FF++ dataset.
    """
    def __init__(self, pretrained=True):
        super(XceptionNet, self).__init__()
        self.xception_base = pretrainedmodels.xception(pretrained='imagenet' if pretrained else None)
        num_features = self.xception_base.last_linear.in_features
        self.xception_base.last_linear = nn.Linear(num_features, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        features = self.xception_base(x)
        return self.sigmoid(features)
    
    def process_video(self, video_frames):
        batch_size, num_frames = video_frames.shape[:2]
        frames_reshaped = video_frames.view(-1, 3, video_frames.shape[3], video_frames.shape[4])
        frame_preds = self.forward(frames_reshaped)
        frame_preds = frame_preds.view(batch_size, num_frames, -1)
        return torch.mean(frame_preds, dim=1), frame_preds

class LipSyncDetector(nn.Module):
    """Implementation of a lip-sync detection model for deepfake detection"""
    def __init__(self):
        super(LipSyncDetector, self).__init__()
        self.visual_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.visual_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.visual_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.visual_pool = nn.MaxPool2d(2, 2)
        self.audio_conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.audio_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.audio_pool = nn.MaxPool1d(2, 2)
        self.visual_features = 256 * 32 * 32
        self.audio_features = 128 * 32
        self.fusion_fc1 = nn.Linear(self.visual_features + self.audio_features, 512)
        self.fusion_fc2 = nn.Linear(512, 128)
        self.fusion_fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, visual, audio):
        v = F.relu(self.visual_conv1(visual))
        v = self.visual_pool(v)
        v = F.relu(self.visual_conv2(v))
        v = self.visual_pool(v)
        v = F.relu(self.visual_conv3(v))
        v = self.visual_pool(v)
        v = v.view(v.size(0), -1)
        a = F.relu(self.audio_conv1(audio))
        a = self.audio_pool(a)
        a = F.relu(self.audio_conv2(a))
        a = self.audio_pool(a)
        a = a.view(a.size(0), -1)
        fusion = torch.cat((v, a), dim=1)
        fusion = F.relu(self.fusion_fc1(fusion))
        fusion = self.dropout(fusion)
        fusion = F.relu(self.fusion_fc2(fusion))
        fusion = self.dropout(fusion)
        return torch.sigmoid(self.fusion_fc3(fusion))
    
    def process_video(self, video_frames, audio_features):
        if video_frames.dim() != 5:
            raise ValueError("video_frames must have shape (batch, frames, channels, height, width)")

        visual = video_frames.mean(dim=1)
        if visual.shape[-2:] != (256, 256):
            visual = F.interpolate(visual, size=(256, 256), mode='bilinear', align_corners=False)

        audio = self._prepare_audio(audio_features)
        predictions = self.forward(visual, audio)
        attention_weights = torch.ones(predictions.shape[0], 2, device=predictions.device)
        return predictions, attention_weights

    def _prepare_audio(self, audio_features):
        if audio_features.dim() == 4:
            audio = audio_features.squeeze(1)
        elif audio_features.dim() == 3:
            # Treat the longer axis as time and the shorter as features, which
            # matches both (batch, features, time) and (batch, time, features)
            # inputs produced by the dataset and extract_audio_features.
            if audio_features.shape[1] <= audio_features.shape[2]:
                audio = audio_features
            else:
                audio = audio_features.transpose(1, 2)
        elif audio_features.dim() == 2:
            audio = audio_features.unsqueeze(-1)
        else:
            raise ValueError("audio_features must have 2, 3, or 4 dimensions")

        # Project the feature dimension to the 128 channels expected by the
        # Conv1d stack so MFCC (40), mel (128), or other inputs all work.
        if audio.shape[1] != 128:
            audio = audio.transpose(1, 2)
            audio = F.interpolate(audio, size=128, mode='linear', align_corners=False)
            audio = audio.transpose(1, 2)

        if audio.shape[-1] != 128:
            audio = F.interpolate(audio, size=128, mode='linear', align_corners=False)
        return audio
