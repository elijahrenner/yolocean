# tests/test_utils.py

import unittest
import numpy as np
import cv2
import os
from src.utils import write_polygon_file

class TestUtils(unittest.TestCase):
    def test_write_polygon_file(self):
        # Setup
        class_contour_mapping = {
            1: [np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)],
            2: [np.array([[[30, 30]], [[40, 30]], [[40, 40]], [[30, 40]]], dtype=np.int32)]
        }
        H, W = 100, 100
        output_path = 'tests/output'
        os.makedirs(output_path, exist_ok=True)
        img_name = 'test_image'

        # Expected content
        expected_content = (
            "1 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
            "2 0.3 0.3 0.4 0.3 0.4 0.4 0.3 0.4\n"
        )

        # Execute
        write_polygon_file(class_contour_mapping, H, W, output_path, img_name)

        # Verify
        with open(os.path.join(output_path, f"{img_name}.txt"), 'r') as f:
            content = f.read()
            self.assertEqual(content, expected_content)

        # Cleanup
        os.remove(os.path.join(output_path, f"{img_name}.txt"))
        os.rmdir(output_path)

if __name__ == '__main__':
    unittest.main()