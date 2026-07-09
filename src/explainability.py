"""
Grad-CAM explainability for the multimodal deepfake detection model.

Implements Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017)
for the visual spatial backbone, visual frequency backbone, and audio CNN.
This satisfies the title report §3.3.3.3 qualitative error analysis requirement:
"Utilize visualization techniques like Grad-CAM for the visual streams to
understand which regions of the face (e.g., eyes, mouth, cheeks) the model
finds most salient for its decision."

Usage:
    from explainability import GradCAMVisualizer

    visualizer = GradCAMVisualizer(model, device)
    heatmaps = visualizer.generate(video, audio)
    visualizer.save_overlay(heatmaps, frame_idx=0, output_path="gradcam_frame0.png")

CLI:
    python src/explainability.py --model_path results/weights/trained_model_multimodal_best.pth \
        --video_path path/to/video.mp4 --output_dir results/images/gradcam
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.multimodal_model import MultimodalModel


class _FeatureHook:
    """Captures forward (activations) and backward (gradients) for a target layer."""

    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self._fwd_hook = module.register_forward_hook(self._save_activation)
        self._bwd_hook = module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class GradCAMVisualizer:
    """
    Grad-CAM for the MultimodalModel's visual (spatial + frequency) and audio
    streams. Registers hooks on the last convolutional layers, runs a forward
    + backward pass, and produces heatmaps localizing the model's attention.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.eval()
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook the last conv layer of each active stream."""
        self.target_layers = {}

        # Spatial backbone — EfficientNet's conv_head is the last conv before pooling.
        if self.model.visual_stream is not None:
            spatial_target = getattr(self.model.visual_stream.spatial_backbone, 'conv_head', None)
            if spatial_target is None:
                # Fallback: find the last Conv2d in the backbone
                spatial_target = self._find_last_conv(self.model.visual_stream.spatial_backbone)
            if spatial_target is not None:
                self.target_layers['spatial'] = spatial_target

            # Frequency backbone (if active)
            if self.model.visual_stream.frequency_backbone is not None:
                freq_target = getattr(self.model.visual_stream.frequency_backbone, 'conv_head', None)
                if freq_target is None:
                    freq_target = self._find_last_conv(self.model.visual_stream.frequency_backbone)
                if freq_target is not None:
                    self.target_layers['frequency'] = freq_target

        # Audio stream — last Conv2d in the sequential (64 -> 128)
        if self.model.audio_stream is not None:
            audio_target = self._find_last_conv(self.model.audio_stream.conv_layers)
            if audio_target is not None:
                self.target_layers['audio'] = audio_target

        for name, layer in self.target_layers.items():
            hook = _FeatureHook(layer)
            self.hooks.append(hook)
            setattr(self, f'_hook_{name}', hook)

    @staticmethod
    def _find_last_conv(module):
        """Find the last Conv2d layer in a module's children."""
        last_conv = None
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        return last_conv

    def generate(self, video, audio):
        """
        Run a forward + backward pass and compute Grad-CAM heatmaps.

        Args:
            video: (batch, frames, C, H, W) tensor
            audio: (batch, ...) audio features

        Returns:
            dict mapping stream name ('spatial', 'frequency', 'audio') to a
            numpy heatmap. Visual heatmaps have shape (batch, frames, H, W);
            audio heatmap has shape (batch, H_audio, W_audio).
        """
        video = video.to(self.device)
        audio = audio.to(self.device)
        video.requires_grad_(True)

        # Forward pass — get scalar prediction for backward
        output = self.model(video, audio)
        # Use the mean prediction across the batch as the target score
        score = output.mean()

        self.model.zero_grad()
        score.backward(retain_graph=False)

        heatmaps = {}
        for name in self.target_layers:
            hook = getattr(self, f'_hook_{name}')
            if hook.activations is None or hook.gradients is None:
                continue
            cam = self._compute_cam(hook.activations, hook.gradients)
            heatmaps[name] = cam

        return heatmaps

    @staticmethod
    def _compute_cam(activations, gradients):
        """
        Compute Grad-CAM from captured activations and gradients.

        Args:
            activations: (batch, channels, H, W) feature map
            gradients:   (batch, channels, H, W) gradient of score w.r.t. features

        Returns:
            numpy array of shape (batch, H, W) with values in [0, 1].
        """
        # Global-average-pool the gradients -> per-channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
        # Weighted sum of activation maps
        cam = (weights * activations).sum(dim=1)  # (batch, H, W)
        cam = F.relu(cam)
        # Normalize per-sample to [0, 1]
        cam = cam.cpu().detach().numpy()
        results = []
        for i in range(cam.shape[0]):
            sample = cam[i]
            mn, mx = sample.min(), sample.max()
            sample = (sample - mn) / (mx - mn + 1e-8)
            results.append(sample)
        return np.array(results)

    def save_overlay(self, heatmaps, video, frame_idx=0, output_dir="results/images/gradcam"):
        """
        Save Grad-CAM heatmaps overlaid on the input video frames and audio.

        Args:
            heatmaps: dict from self.generate()
            video: (batch, frames, C, H, W) tensor — the input video used for generate()
            frame_idx: which frame to visualize
            output_dir: where to save images
        """
        os.makedirs(output_dir, exist_ok=True)

        # --- Visual heatmaps overlaid on video frame ---
        video_np = video[0].cpu().numpy()  # (frames, C, H, W)
        frame = video_np[frame_idx]  # (C, H, W)
        # Denormalize for display (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        frame_disp = frame * std + mean
        frame_disp = np.clip(frame_disp, 0, 1)
        frame_disp = np.transpose(frame_disp, (1, 2, 0))  # (H, W, C)

        for stream_name in ('spatial', 'frequency'):
            if stream_name not in heatmaps:
                continue
            cam = heatmaps[stream_name]
            # cam is (batch, H_feat, W_feat) — take first sample
            if cam.shape[0] == 0:
                continue
            # For video, the batch dim corresponds to frames after reshaping.
            # The spatial backbone processes (batch*frames) images, so cam has
            # shape (batch*frames, H, W). We pick the requested frame.
            cam_frame = cam[min(frame_idx, len(cam) - 1)]
            # Resize heatmap to frame size
            H, W = frame_disp.shape[:2]
            cam_resized = cv2.resize(cam_frame, (W, H))
            cam_uint8 = np.uint8(255 * cam_resized)
            heatmap_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

            overlay = 0.5 * frame_disp + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(frame_disp)
            axes[0].set_title(f'Input Frame {frame_idx}')
            axes[0].axis('off')
            axes[1].imshow(cam_resized, cmap='jet', vmin=0, vmax=1)
            axes[1].set_title(f'{stream_name.capitalize()} Grad-CAM')
            axes[1].axis('off')
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay')
            axes[2].axis('off')
            plt.tight_layout()
            path = os.path.join(output_dir, f"gradcam_{stream_name}_frame{frame_idx}.png")
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved: {path}")

        # --- Audio heatmap ---
        if 'audio' in heatmaps:
            cam = heatmaps['audio']
            if cam.shape[0] > 0:
                cam_audio = cam[0]
                fig, ax = plt.subplots(figsize=(8, 4))
                im = ax.imshow(cam_audio, cmap='jet', aspect='auto', vmin=0, vmax=1)
                ax.set_title('Audio Stream Grad-CAM')
                ax.set_xlabel('Time')
                ax.set_ylabel('Frequency (MFCC bin)')
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                path = os.path.join(output_dir, "gradcam_audio.png")
                plt.savefig(path, dpi=200, bbox_inches='tight')
                plt.close()
                print(f"Saved: {path}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def _load_model_and_data(args, device):
    """Helper for CLI: load model and a single video sample."""
    model = MultimodalModel(
        pretrained=False,
        streams=tuple(args.streams),
        fusion_type=args.fusion_type,
        use_frequency=args.use_frequency,
    ).to(device)

    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        try:
            model.load_state_dict(state_dict)
            print(f"Loaded model from {args.model_path}")
        except RuntimeError as e:
            print(f"Warning: could not load state dict: {e}")
    else:
        print(f"Model not found at {args.model_path}, using untrained weights.")

    from data_loader.datasets import DeepfakeDataset
    from data_loader.video_processor import VideoProcessor
    from data_loader.audio_processor import AudioProcessor
    import torch as _torch

    # Extract frames and audio from the video
    vp = VideoProcessor()
    ap = AudioProcessor()
    frames = vp.get_frames(args.video_path, num_frames=args.num_frames)
    if not frames:
        print(f"Could not extract frames from {args.video_path}")
        return None, None, None

    face_frames = [vp.crop_face(frame) for frame in frames]
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    video_tensor = _torch.stack([transform(f) for f in face_frames]).unsqueeze(0)  # (1, frames, C, H, W)

    audio_waveform = ap.extract_audio(args.video_path)
    mel = ap.to_mel_spectrogram(audio_waveform)
    audio_tensor = _torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)

    return model, video_tensor, audio_tensor


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM explainability (title report §3.3.3.3)')
    parser.add_argument('--model_path', type=str,
                       default='results/weights/trained_model_multimodal_best.pth')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to a video file to analyze')
    parser.add_argument('--output_dir', type=str, default='results/images/gradcam')
    parser.add_argument('--num_frames', type=int, default=10)
    parser.add_argument('--frame_idx', type=int, default=0,
                       help='Which frame to visualize')
    parser.add_argument('--fusion_type', type=str, default='attention',
                       choices=['attention', 'concat'])
    parser.add_argument('--use_frequency', action='store_true', default=True)
    parser.add_argument('--no_frequency', dest='use_frequency', action='store_false')
    parser.add_argument('--streams', type=str, nargs='+',
                       default=['visual', 'audio', 'temporal'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, video, audio = _load_model_and_data(args, device)
    if model is None:
        return

    print(f"Video shape: {video.shape}")
    print(f"Audio shape: {audio.shape}")

    visualizer = GradCAMVisualizer(model, device)
    heatmaps = visualizer.generate(video, audio)
    visualizer.save_overlay(heatmaps, video, frame_idx=args.frame_idx, output_dir=args.output_dir)
    visualizer.remove_hooks()

    print(f"\nGrad-CAM visualizations saved to {args.output_dir}/")
    print("Streams visualized:", list(heatmaps.keys()))


if __name__ == '__main__':
    main()
