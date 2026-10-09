"""Unit tests for character segmentation, visual overlay, and per-stage anti-blur."""

import unittest
import numpy as np
import cv2

from detect_plate import (
    deblur_vehicle_stage,
    deblur_plate_stage,
    deblur_char_stage,
    draw_character_annotations,
    TwoStagePlateDetector,
)


class AntiBlurTests(unittest.TestCase):
    def test_deblur_vehicle_stage(self):
        img = np.ones((100, 150, 3), dtype=np.uint8) * 120
        res = deblur_vehicle_stage(img)
        self.assertEqual(res.shape, (100, 150, 3))
        self.assertEqual(res.dtype, np.uint8)

    def test_deblur_plate_stage(self):
        crop = np.ones((50, 80, 3), dtype=np.uint8) * 100
        res = deblur_plate_stage(crop)
        self.assertEqual(res.shape, (50, 80, 3))
        self.assertEqual(res.dtype, np.uint8)

    def test_deblur_char_stage(self):
        plate = np.ones((40, 120, 3), dtype=np.uint8) * 140
        # Draw some dark strokes
        cv2.putText(plate, "KA09", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
        res = deblur_char_stage(plate)
        self.assertEqual(res.shape, (40, 120, 3))
        self.assertEqual(res.dtype, np.uint8)


class CharacterSegmentationTests(unittest.TestCase):
    def setUp(self):
        # Create a synthetic white license plate with dark characters
        self.plate = np.ones((40, 140, 3), dtype=np.uint8) * 240
        # Draw black border
        cv2.rectangle(self.plate, (2, 2), (137, 37), (0, 0, 0), 1)
        # Draw distinct characters
        chars = ["K", "A", "0", "9"]
        for idx, ch in enumerate(chars):
            x = 20 + idx * 28
            cv2.putText(self.plate, ch, (x, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    def test_draw_character_annotations(self):
        chars = [
            {"char_id": 1, "char_text": "K", "confidence": 0.9, "bbox_plate_xyxy": [18, 10, 35, 30]},
            {"char_id": 2, "char_text": "A", "confidence": 0.9, "bbox_plate_xyxy": [46, 10, 63, 30]},
        ]
        ann = draw_character_annotations(self.plate, chars)
        self.assertIsNotNone(ann)
        self.assertGreater(ann.shape[0], self.plate.shape[0])  # Scaled for crisp display
        self.assertEqual(ann.shape[2], 3)

    def test_segment_characters_mock_detector(self):
        # Detector instance without loading heavy models for fast unit test
        detector = object.__new__(TwoStagePlateDetector)
        detector.ocr_reader = None
        characters, annotated = TwoStagePlateDetector.extract_character_segments(
            detector,
            self.plate,
            recognized_text="KA 09",
            ocr_confidence=0.92,
        )
        self.assertIsInstance(characters, list)
        self.assertGreaterEqual(len(characters), 1)
        first_char = characters[0]
        self.assertIn("char_id", first_char)
        self.assertIn("bbox_plate_xyxy", first_char)
        self.assertIn("char_crop_bgr", first_char)
        self.assertGreater(first_char["char_crop_bgr"].shape[0], 0)
        self.assertIsNotNone(annotated)


if __name__ == "__main__":
    unittest.main()
