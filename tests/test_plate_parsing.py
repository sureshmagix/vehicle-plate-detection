"""Regression tests for conservative registration-number parsing."""

import unittest

from detect_plate import clean_and_parse_plate


class PlateParsingTests(unittest.TestCase):
    def test_formats_standard_indian_plate(self):
        self.assertEqual(clean_and_parse_plate("KA09MB7389")[0], "KA 09 MB 7389")
        self.assertEqual(clean_and_parse_plate("KA012345")[0], "KA 01 2345")

    def test_corrects_lookalikes_only_in_known_fields(self):
        self.assertEqual(clean_and_parse_plate("K409MB7389")[0], "KA 09 MB 7389")
        self.assertEqual(clean_and_parse_plate("KAO9MB7389")[0], "KA 09 MB 7389")

    def test_formats_bharat_series(self):
        self.assertEqual(clean_and_parse_plate("22BH1234AA")[0], "22 BH 1234 AA")

    def test_does_not_invent_a_state_from_a_partial_match(self):
        text, score = clean_and_parse_plate("LOS4B345")
        self.assertEqual(text, "LOS 4B 345")
        self.assertLess(score, 0.8)


if __name__ == "__main__":
    unittest.main()
