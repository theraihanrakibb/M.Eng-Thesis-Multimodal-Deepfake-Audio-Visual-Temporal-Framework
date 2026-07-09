import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class MediaProcessor:
    """
    Class to handle processing of video and audio files for deepfake detection
    """
    
    def __init__(self, media_dir="media"):
        self.media_dir = Path(media_dir)
        self.real_videos_dir = self.media_dir / "real" / "videos"
        self.real_audio_dir = self.media_dir / "real" / "audio"
        self.fake_videos_dir = self.media_dir / "fake" / "videos"
        self.fake_audio_dir = self.media_dir / "fake" / "audio"
        self.gifs_dir = self.media_dir / "gifs"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create media directories if they don't exist"""
        self.real_videos_dir.mkdir(parents=True, exist_ok=True)
        self.real_audio_dir.mkdir(parents=True, exist_ok=True)
        self.fake_videos_dir.mkdir(parents=True, exist_ok=True)
        self.fake_audio_dir.mkdir(parents=True, exist_ok=True)
        self.gifs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_media_files(self):
        """
        Get all media files from the media directory
        
        Returns:
            dict: Dictionary containing lists of media files by category
        """
        media_files = {
            "real_videos": list(self.real_videos_dir.glob("*.mp4")) + list(self.real_videos_dir.glob("*.avi")) + list(self.real_videos_dir.glob("*.mov")),
            "real_audio": list(self.real_audio_dir.glob("*.wav")) + list(self.real_audio_dir.glob("*.mp3")),
            "fake_videos": list(self.fake_videos_dir.glob("*.mp4")) + list(self.fake_videos_dir.glob("*.avi")) + list(self.fake_videos_dir.glob("*.mov")),
            "fake_audio": list(self.fake_audio_dir.glob("*.wav")) + list(self.fake_audio_dir.glob("*.mp3")),
            "gifs": list(self.gifs_dir.glob("*.gif"))
        }
        
        return media_files
    
    def process_video_file(self, video_path, num_frames=10):
        """
        Process a video file and extract frames
        
        Args:
            video_path (str): Path to video file
            num_frames (int): Number of frames to extract
            
        Returns:
            torch.Tensor: Extracted video frames
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"Warning: Video {video_path} has no frames")
                return None
            
            # Select frames to extract
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (256, 256))
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                print(f"Warning: Could not extract frames from {video_path}")
                return None
            
            # Convert to tensor
            frames = np.array(frames)
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
            frames = frames / 255.0  # Normalize
            
            return frames
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return None
    
    def process_audio_file(self, audio_path, num_segments=10):
        """
        Process an audio file and extract features
        
        Args:
            audio_path (str): Path to audio file
            num_segments (int): Number of segments to extract
            
        Returns:
            torch.Tensor: Extracted audio features
        """
        # Placeholder for audio processing
        # In a real implementation, you would use librosa or similar
        # to extract MFCCs or other audio features
        
        try:
            # For now, return random features
            features = torch.rand(num_segments, 128)
            return features
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return None
    
    def create_comparison_gif(self, real_video_path, fake_video_path, output_path):
        """
        Create a comparison GIF showing real vs fake video frames
        
        Args:
            real_video_path (str): Path to real video
            fake_video_path (str): Path to fake video
            output_path (str): Output path for GIF
        """
        try:
            # Extract frames from both videos
            real_frames = self.process_video_file(real_video_path, num_frames=5)
            fake_frames = self.process_video_file(fake_video_path, num_frames=5)
            
            if real_frames is None or fake_frames is None:
                print("Could not create comparison GIF - missing frames")
                return
            
            # Create comparison images
            comparison_images = []
            
            for i in range(min(len(real_frames), len(fake_frames))):
                # Convert tensors to numpy arrays
                real_frame = real_frames[i].permute(1, 2, 0).numpy() * 255
                fake_frame = fake_frames[i].permute(1, 2, 0).numpy() * 255
                
                # Create side-by-side comparison
                comparison = np.hstack([real_frame, fake_frame])
                comparison = comparison.astype(np.uint8)
                
                # Add labels
                comparison_with_text = cv2.putText(
                    comparison.copy(), 
                    "Real", 
                    (50, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )
                comparison_with_text = cv2.putText(
                    comparison_with_text, 
                    "Fake", 
                    (306, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 255, 255), 
                    2
                )
                
                comparison_images.append(Image.fromarray(comparison_with_text))
            
            # Save as GIF
            if comparison_images:
                comparison_images[0].save(
                    output_path,
                    save_all=True,
                    append_images=comparison_images[1:],
                    duration=500,  # 500ms per frame
                    loop=0  # Loop indefinitely
                )
                print(f"Comparison GIF saved to {output_path}")
            
        except Exception as e:
            print(f"Error creating comparison GIF: {e}")
    
    def generate_sample_media(self):
        """
        Generate sample media files for testing
        """
        print("Generating sample media files...")
        
        # Create sample text files that can be replaced with actual media
        sample_files = [
            (self.real_videos_dir / "sample_real_video.mp4.txt", "Placeholder for real video content"),
            (self.real_audio_dir / "sample_real_audio.wav.txt", "Placeholder for real audio content"),
            (self.fake_videos_dir / "sample_fake_video.mp4.txt", "Placeholder for fake video content"),
            (self.fake_audio_dir / "sample_fake_audio.wav.txt", "Placeholder for fake audio content"),
            (self.gifs_dir / "real_vs_fake_comparison.gif.txt", "Placeholder for GIF comparison")
        ]
        
        for file_path, content in sample_files:
            with open(file_path, 'w') as f:
                f.write(content)
        
        print("Sample media files created. Replace .txt files with actual media files.")

# Example usage
if __name__ == "__main__":
    processor = MediaProcessor()
    processor.generate_sample_media()
    
    # Get media files
    media_files = processor.get_media_files()
    print("\nMedia files found:")
    for category, files in media_files.items():
        print(f"{category}: {len(files)} files")
