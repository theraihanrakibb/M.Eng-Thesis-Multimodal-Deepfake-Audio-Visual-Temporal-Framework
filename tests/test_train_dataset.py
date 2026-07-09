import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("src"))

from train import create_dataset


class TestTrainDataset(unittest.TestCase):

    def test_create_dataset_uses_real_fake_folder_labels(self):
        data_dir = "dataset"

        def fake_walk(path):
            if path.endswith("real"):
                return [(path, [], ["sample_real.mp4"])]
            if path.endswith("fake"):
                return [(path, [], ["sample_fake.mp4"])]
            return []

        with patch("train.os.path.exists", return_value=True), \
             patch("train.os.path.isdir", return_value=True), \
             patch("train.os.walk", side_effect=fake_walk):
            dataset = create_dataset(data_dir)

        self.assertEqual(sorted(dataset.labels), [0, 1])
        self.assertEqual(len(dataset.video_paths), 2)


if __name__ == '__main__':
    unittest.main()
