import unittest
import torch
from src.models.multimodal_model import VisualStream, AudioStream, TemporalStream, MultimodalModel

class TestModels(unittest.TestCase):

    def test_visual_stream(self):
        model = VisualStream(pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        expected_dim = model.feature_dim * 2
        self.assertEqual(output.shape, (2, expected_dim))

    def test_audio_stream(self):
        model = AudioStream(feature_dim=512)
        dummy_input = torch.randn(2, 1, 128, 200)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 512))

    def test_temporal_stream(self):
        visual_feature_dim = 1280 * 2
        model = TemporalStream(input_dim=visual_feature_dim)
        dummy_input = torch.randn(2, 30, visual_feature_dim)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, visual_feature_dim))

    def test_multimodal_model(self):
        model = MultimodalModel(pretrained=False)
        dummy_video = torch.randn(2, 30, 3, 224, 224)
        dummy_audio = torch.randn(2, 1, 128, 200)
        output = model(dummy_video, dummy_audio)
        self.assertEqual(output.shape, (2, 1))

if __name__ == '__main__':
    unittest.main()
