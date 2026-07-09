import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VisualStream(nn.Module):
    def __init__(self, model_name='efficientnet_b0', pretrained=True, use_frequency=True):
        """
        Visual stream extracting spatial and optionally frequency-domain features.

        Args:
            use_frequency: If True (default), include the FFT frequency branch
                and concatenate spatial + frequency features. If False, return
                only spatial features. Used for input-representation ablation
                (title report §3.3.2.2).
        """
        super(VisualStream, self).__init__()
        self.spatial_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.spatial_backbone.num_features
        self.use_frequency = use_frequency
        if use_frequency:
            self.frequency_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, in_chans=1)
            self.output_dim = self.feature_dim * 2
        else:
            self.frequency_backbone = None
            self.output_dim = self.feature_dim

    def forward(self, x):
        spatial_features = self.spatial_backbone(x)
        if not self.use_frequency:
            return spatial_features
        x_gray = torch.mean(x, dim=1, keepdim=True)
        fft_out = torch.fft.fft2(x_gray)
        fft_magnitude = torch.abs(fft_out)
        fft_magnitude = torch.log1p(fft_magnitude)
        fft_magnitude = (fft_magnitude - fft_magnitude.mean()) / (fft_magnitude.std() + 1e-6)
        frequency_features = self.frequency_backbone(fft_magnitude)
        return torch.cat((spatial_features, frequency_features), dim=1)

class AudioStream(nn.Module):
    def __init__(self, in_channels=1, feature_dim=512, use_mfcc=True):
        """
        Audio stream for processing MFCC or mel spectrogram features.

        Args:
            in_channels: Input channels (1 for mono audio features)
            feature_dim: Output feature dimension
            use_mfcc: If True, expects MFCC input (40 coeffs), else mel spectrogram (128 bands)
        """
        super(AudioStream, self).__init__()
        self.use_mfcc = use_mfcc
        self.input_features = 40 if use_mfcc else 128

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class TemporalStream(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=3, dropout=0.1):
        super(TemporalStream, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.transformer_encoder(x)
        return torch.mean(x, dim=1)

class AttentionFusion(nn.Module):
    """
    Attention-based fusion over a variable number of streams. Each stream's
    features are projected to fusion_dim, then a per-stream attention weight
    is computed and applied.
    """
    def __init__(self, stream_dims, fusion_dim=512):
        super(AttentionFusion, self).__init__()
        self.stream_names = list(stream_dims.keys())
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fusion_dim) for name, dim in stream_dims.items()
        })
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim, max(fusion_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(fusion_dim // 2, 1), 1)
        )
        self.last_attention_weights = None

    def forward(self, features_dict):
        projected = {name: self.projections[name](features_dict[name]) for name in self.stream_names}
        stacked = torch.stack([projected[name] for name in self.stream_names], dim=1)
        weights = torch.softmax(self.attention(stacked).squeeze(-1), dim=1)
        self.last_attention_weights = weights
        weighted = stacked * weights.unsqueeze(-1)
        return torch.sum(weighted, dim=1)

class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion — baseline for fusion-mechanism ablation
    (title report §3.3.2.2). Concatenates all stream features and projects
    through a single linear + ReLU layer.
    """
    def __init__(self, stream_dims, fusion_dim=512):
        super(ConcatFusion, self).__init__()
        self.stream_names = list(stream_dims.keys())
        total_dim = sum(stream_dims.values())
        self.projection = nn.Linear(total_dim, fusion_dim)
        self.relu = nn.ReLU()
        self.last_attention_weights = None

    def forward(self, features_dict):
        concatenated = torch.cat([features_dict[name] for name in self.stream_names], dim=1)
        return self.relu(self.projection(concatenated))

class MultimodalModel(nn.Module):
    def __init__(self, num_classes=1, visual_model_name='efficientnet_b0',
                 audio_feature_dim=512, pretrained=True, use_mfcc=True,
                 streams=('visual', 'audio', 'temporal'),
                 fusion_type='attention',
                 use_frequency=True):
        """
        Multimodal deepfake detection model with configurable streams and fusion.

        Args:
            num_classes: Number of output classes (1 for binary real/fake)
            visual_model_name: Backbone model name for visual stream
            audio_feature_dim: Dimension of audio feature vector (also the fusion_dim)
            pretrained: Use pretrained backbones
            use_mfcc: Use MFCC features (True for thesis, default) or mel spectrogram
            streams: Tuple of stream names to use. Any subset of
                ('visual', 'audio', 'temporal'). The temporal stream internally
                uses the visual backbone for frame features, so 'temporal'
                requires the visual backbone to be instantiated (but 'visual'
                features need not be in the fusion).
            fusion_type: 'attention' (default) or 'concat'
            use_frequency: If True, visual stream includes the FFT frequency
                branch. If False, spatial-only. Used for input-representation
                ablation (title report §3.3.2.2).
        """
        super(MultimodalModel, self).__init__()
        self.streams = tuple(streams)
        self.use_frequency = use_frequency
        self.use_mfcc = use_mfcc
        self.audio_feature_dim = audio_feature_dim

        # The visual backbone is needed whenever 'visual' or 'temporal' is active,
        # because the temporal stream consumes frame-level visual features.
        needs_visual_backbone = 'visual' in self.streams or 'temporal' in self.streams
        if needs_visual_backbone:
            self.visual_stream = VisualStream(
                model_name=visual_model_name, pretrained=pretrained, use_frequency=use_frequency
            )
            visual_feature_dim = self.visual_stream.feature_dim * (2 if use_frequency else 1)
        else:
            self.visual_stream = None
            visual_feature_dim = None

        if 'audio' in self.streams:
            self.audio_stream = AudioStream(feature_dim=audio_feature_dim, use_mfcc=use_mfcc)
        else:
            self.audio_stream = None

        if 'temporal' in self.streams:
            self.temporal_stream = TemporalStream(input_dim=visual_feature_dim)
        else:
            self.temporal_stream = None

        # Build fusion with the dims of the active streams.
        stream_dims = {}
        if 'visual' in self.streams:
            stream_dims['visual'] = visual_feature_dim
        if 'audio' in self.streams:
            stream_dims['audio'] = audio_feature_dim
        if 'temporal' in self.streams:
            stream_dims['temporal'] = visual_feature_dim

        if fusion_type == 'attention':
            self.fusion = AttentionFusion(stream_dims, fusion_dim=audio_feature_dim)
        elif fusion_type == 'concat':
            self.fusion = ConcatFusion(stream_dims, fusion_dim=audio_feature_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Use 'attention' or 'concat'.")

        self.classifier = nn.Sequential(
            nn.Linear(audio_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        features = {}
        frame_features = None

        # Extract frame-level visual features if visual or temporal stream is active.
        if self.visual_stream is not None:
            batch_size, num_frames = video.shape[:2]
            video_reshaped = video.view(-1, *video.shape[2:])
            frame_features = self.visual_stream(video_reshaped)
            frame_features = frame_features.view(batch_size, num_frames, -1)

        if 'visual' in self.streams:
            features['visual'] = frame_features.mean(dim=1)

        if 'audio' in self.streams:
            if len(audio.shape) == 3:
                audio = audio.unsqueeze(1)
            elif len(audio.shape) == 4 and audio.shape[1] != 1:
                audio = audio[:, 0:1, :, :]
            features['audio'] = self.audio_stream(audio)

        if 'temporal' in self.streams:
            features['temporal'] = self.temporal_stream(frame_features)

        fused_features = self.fusion(features)
        return self.classifier(fused_features)

    def process_video(self, video_frames, audio_features):
        predictions = self.forward(video_frames, audio_features)
        batch_size = predictions.shape[0]
        attention_weights = self.fusion.last_attention_weights
        if attention_weights is None:
            num_streams = len(self.streams)
            attention_weights = torch.full(
                (batch_size, num_streams), 1.0 / num_streams, device=predictions.device
            )
        return predictions, attention_weights
