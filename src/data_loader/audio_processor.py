import numpy as np
from moviepy import VideoFileClip
import librosa

def _fix_audio_length(audio, target_samples):
    if len(audio) >= target_samples:
        return audio[:target_samples]
    return np.pad(audio, (0, target_samples - len(audio)))

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(sr, n_fft, n_mels):
    mel_min = _hz_to_mel(0)
    mel_max = _hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return filters

def extract_audio_features(video_path, sr=16000, n_mels=128, n_fft=2048, hop_length=512, use_mfcc=True):
    """
    Extract audio features from a video file.
    Returns a log-mel spectrogram or MFCC features with a stable shape for batching.

    Args:
        use_mfcc: If True (default), return MFCC features (n_mfcc x time).
                  If False, return a mel spectrogram (n_mels x time).
    """
    processor = AudioProcessor(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_mfcc=use_mfcc)
    audio = processor.extract_audio(video_path)
    return processor.to_mel_spectrogram(audio)

def _waveform_to_log_mel(audio_waveform, sr, n_fft, hop_length, n_mels):
    padded = np.pad(audio_waveform, (n_fft // 2, n_fft // 2), mode="constant")
    frame_count = 1 + (len(padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop_length][:frame_count]
    windowed = frames * np.hanning(n_fft)
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).T
    mel_spectrogram = _mel_filterbank(sr, n_fft, n_mels) @ power

    amin = 1e-10
    log_mel = 10.0 * np.log10(np.maximum(mel_spectrogram, amin))
    log_mel -= 10.0 * np.log10(max(np.max(mel_spectrogram), amin))
    return log_mel.astype(np.float32)

class AudioProcessor:
    def __init__(self, sr=16000, n_fft=2048, hop_length=512, n_mels=128, duration=5.0, use_mfcc=True, n_mfcc=40):
        """
        Audio processor with MFCC feature extraction.
        
        Args:
            sr: Sample rate (default: 16000 Hz)
            n_fft: FFT window size (default: 2048)
            hop_length: Hop length for STFT (default: 512)
            n_mels: Number of mel bands (default: 128)
            duration: Audio duration to process (default: 5.0 seconds)
            use_mfcc: Use MFCC features instead of mel spectrogram (default: True for thesis)
            n_mfcc: Number of MFCC coefficients (default: 40, standard for speech/deepfake analysis)
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration
        self.target_samples = int(sr * duration)
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc

    def extract_audio(self, video_path):
        """
        Extracts audio waveform from a video file.
        Videos without an audio track return silence so dataset loading can continue.
        """
        try:
            with VideoFileClip(video_path) as clip:
                if clip.audio is None:
                    return np.zeros(self.target_samples, dtype=np.float32)
                audio = clip.audio.to_soundarray(fps=self.sr)
        except Exception as e:
            print(f"Warning: Could not extract audio from {video_path}: {e}. Returning silence.")
            return np.zeros(self.target_samples, dtype=np.float32)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return _fix_audio_length(audio, self.target_samples).astype(np.float32)

    def to_mel_spectrogram(self, audio_waveform):
        """Convert audio waveform to mel spectrogram or MFCC features"""
        if audio_waveform is None or len(audio_waveform) == 0:
            audio_waveform = np.zeros(self.target_samples, dtype=np.float32)

        try:
            audio_waveform = np.asarray(audio_waveform, dtype=np.float32)
            audio_waveform = _fix_audio_length(audio_waveform, self.target_samples)
            
            # Use Librosa MFCC extraction (thesis requirement)
            if self.use_mfcc:
                # Extract MFCC features using Librosa (standard for deepfake audio detection)
                mfcc = librosa.feature.mfcc(
                    y=audio_waveform,
                    sr=self.sr,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
                # Normalize MFCC features (standard practice)
                mfcc_normalized = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
                return mfcc_normalized.astype(np.float32)
            else:
                # Fallback to custom mel spectrogram
                return _waveform_to_log_mel(
                    audio_waveform,
                    sr=self.sr,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels
                )
        except Exception as e:
            print(f"Error converting to audio features: {e}")
            if self.use_mfcc:
                return np.zeros((self.n_mfcc, 1 + self.target_samples // self.hop_length), dtype=np.float32)
            else:
                return np.zeros((self.n_mels, 1 + self.target_samples // self.hop_length), dtype=np.float32)
