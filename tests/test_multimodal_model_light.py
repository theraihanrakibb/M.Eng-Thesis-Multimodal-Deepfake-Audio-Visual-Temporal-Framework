import unittest
from unittest.mock import patch, MagicMock
import torch
from src.models.multimodal_model import MultimodalModel

class TestMultimodalModelLight(unittest.TestCase):

    @patch('src.models.multimodal_model.TemporalStream')
    @patch('src.models.multimodal_model.AudioStream')
    @patch('src.models.multimodal_model.VisualStream')
    def test_multimodal_model_forward_pass(self, MockVisualStream, MockAudioStream, MockTemporalStream):
        # Configure the mock classes to return mock instances
        mock_visual_instance = MagicMock()
        MockVisualStream.return_value = mock_visual_instance
        # Set the feature_dim attribute on the mock instance
        mock_visual_instance.feature_dim = 1280

        mock_audio_instance = MagicMock()
        MockAudioStream.return_value = mock_audio_instance

        mock_temporal_instance = MagicMock()
        MockTemporalStream.return_value = mock_temporal_instance

        # Define the return values of the forward passes of the mock instances
        visual_feature_dim = 1280 * 2
        mock_visual_instance.return_value = torch.randn(2 * 30, visual_feature_dim)

        audio_feature_dim = 512
        # The audio stream is followed by a linear layer to match the visual dim
        mock_audio_instance.return_value = torch.randn(2, audio_feature_dim)

        mock_temporal_instance.return_value = torch.randn(2, visual_feature_dim)

        # Instantiate the model (with mocked sub-modules)
        model = MultimodalModel(audio_feature_dim=audio_feature_dim)

        # Create dummy input tensors
        dummy_video = torch.randn(2, 30, 3, 224, 224)
        dummy_audio = torch.randn(2, 1, 128, 200)

        # Run the forward pass
        output = model(dummy_video, dummy_audio)

        # Assert that the output shape is correct
        self.assertEqual(output.shape, (2, 1))

        # Assert that the sub-modules were called
        MockVisualStream.assert_called_once()
        MockAudioStream.assert_called_once()
        MockTemporalStream.assert_called_once()
        self.assertEqual(mock_temporal_instance.call_args.args[0].shape, (2, 30, visual_feature_dim))

        with patch('src.models.multimodal_model.torch.randn', side_effect=AssertionError("random predictions are not allowed")):
            predictions, attention_weights = model.process_video(dummy_video, dummy_audio)

        self.assertEqual(predictions.shape, (2, 1))
        self.assertEqual(attention_weights.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()
