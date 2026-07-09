import cv2
import torch.nn.functional as F
import numpy as np
import os

def extract_frames(video_path, num_frames=30):
    """Extracts a specified number of frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def preprocess_frames(frames, target_size=(224, 224)):
    """Preprocesses frames for model input."""
    processed_frames = []
    for frame in frames:
        # Resize frame
        resized = cv2.resize(frame, target_size)
        # Convert to RGB (from BGR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize
        normalized = rgb_frame / 255.0
        processed_frames.append(normalized)
    
    return np.array(processed_frames)

class VideoProcessor:
    def __init__(self, face_cascade_path=None):
        if face_cascade_path is None:
            # Look for the default path for the haarcascade file
            # This is a common location in many environments
            cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
            face_cascade_path = os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml')

        if not os.path.exists(face_cascade_path):
            import warnings
            warnings.warn(f"Face cascade file not found at: {face_cascade_path}. Face cropping disabled.")
            self.face_cascade = None
        else:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def get_frames(self, video_path, num_frames=30):
        """Extracts a specified number of frames from a video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return []

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def crop_face(self, frame, size=(224, 224)):
        """Crops the face from a single frame."""
        if self.face_cascade is None:
            return cv2.resize(frame, size)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Assume the largest face is the one of interest
            x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face = frame[y:y+h, x:x+w]
            return cv2.resize(face, size)
        else:
            # If no face is found, return the resized frame
            return cv2.resize(frame, size)
