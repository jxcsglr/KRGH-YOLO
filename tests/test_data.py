import tempfile
import unittest
from pathlib import Path

from krgh_yolo.data import resolve_data


class TestDatasetConfig(unittest.TestCase):
    def test_directory_dataset_generates_yaml(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset"
            for relative in (
                "images/train",
                "images/val",
                "labels/train",
                "labels/val",
            ):
                (root / relative).mkdir(parents=True)
            info = resolve_data(root, Path(temp_dir) / "configs", preset="scb")
            self.assertTrue(info["yaml"].is_file())
            self.assertEqual(len(info["class_names"]), 6)
            self.assertEqual(info["counts"]["train_images"], 0)


if __name__ == "__main__":
    unittest.main()
