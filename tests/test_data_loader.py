import unittest
from unittest.mock import patch
import os
import numpy as np
import torch
import soundfile as sf

from src.data_loader.video_processor import VideoProcessor
from src.data_loader.audio_processor import AudioProcessor
from src.data_loader.datasets import DeepfakeDataset

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.video_path = "tests/test_data/dummy.mp4"
        self.audio_path = "tests/test_data/dummy.wav"
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def test_video_processor(self):
        frames = self.video_processor.get_frames(self.video_path, num_frames=10)
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0].shape, (224, 224, 3))

    def test_audio_processor(self):
        waveform, sr = sf.read(self.audio_path)
        # Resample if necessary to match the processor's sample rate
        if sr != self.audio_processor.sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.audio_processor.sr)

        mel_spec = self.audio_processor.to_mel_spectrogram(waveform)
        # AudioProcessor defaults to MFCC features (use_mfcc=True), so the
        # leading dimension is n_mfcc (40). With use_mfcc=False it is n_mels (128).
        expected_features = self.audio_processor.n_mfcc if self.audio_processor.use_mfcc else self.audio_processor.n_mels
        self.assertEqual(mel_spec.shape[0], expected_features)

    def test_audio_processor_is_deterministic(self):
        waveform, sr = sf.read(self.audio_path)
        if sr != self.audio_processor.sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.audio_processor.sr)

        first = self.audio_processor.to_mel_spectrogram(waveform)
        second = self.audio_processor.to_mel_spectrogram(waveform)

        np.testing.assert_allclose(first, second)

    @patch('src.data_loader.audio_processor.AudioProcessor.extract_audio')
    def test_dataset(self, mock_extract_audio):
        dummy_waveform, sr = sf.read(self.audio_path)
        if sr != self.audio_processor.sr:
            import librosa
            dummy_waveform = librosa.resample(dummy_waveform, orig_sr=sr, target_sr=self.audio_processor.sr)
        mock_extract_audio.return_value = dummy_waveform

        dataset = DeepfakeDataset(
            video_paths=[self.video_path],
            labels=[0],
            num_frames=15
        )
        sample = dataset[0]

        self.assertIsInstance(sample["video"], torch.Tensor)
        self.assertEqual(sample["video"].shape, (15, 3, 224, 224))

        self.assertIsInstance(sample["audio"], torch.Tensor)
        self.assertEqual(sample["audio"].shape[0], 1) # Channel dimension
        # AudioProcessor defaults to MFCC (use_mfcc=True) -> 40 coefficients;
        # mel spectrogram (use_mfcc=False) -> 128 mel bands.
        expected_features = self.audio_processor.n_mfcc if self.audio_processor.use_mfcc else self.audio_processor.n_mels
        self.assertEqual(sample["audio"].shape[1], expected_features)

        self.assertIsInstance(sample["label"], torch.Tensor)
        self.assertEqual(sample["label"], 0)
        mock_extract_audio.assert_called_once_with(self.video_path)


if __name__ == '__main__':
    unittest.main()
