import unittest
import numpy as np
from main import segment_image, correct_illumination, integrate_segments

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        self.image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        self.segment = np.random.randint(0, 256, (25, 25), dtype=np.uint8)

    def test_segment_image_returns_correct_number_of_segments(self):
        segments = segment_image(self.image, num_segments=4)
        self.assertEqual(len(segments), 16)

    def test_segment_image_returns_correct_segment_size(self):
        segments = segment_image(self.image, num_segments=4)
        for segment in segments:
            self.assertEqual(segment.shape, (25, 25))

    def test_correct_illumination_returns_correct_dtype(self):
        corrected_segment = correct_illumination(self.segment)
        self.assertEqual(corrected_segment.dtype, np.uint8)

    def test_integrate_segments_returns_correct_size(self):
        segments = [self.segment] * 16
        integrated_image = integrate_segments(segments, num_segments=4)
        self.assertEqual(integrated_image.shape, (100, 100))

    def test_integrate_segments_returns_correct_dtype(self):
        segments = [self.segment] * 16
        integrated_image = integrate_segments(segments, num_segments=4)
        self.assertEqual(integrated_image.dtype, np.uint8)

    def test_segment_image_with_one_segment(self):
        segments = segment_image(self.image, num_segments=1)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].shape, self.image.shape)

    def test_integrate_segments_with_one_segment(self):
        segments = [self.image]
        integrated_image = integrate_segments(segments, num_segments=1)
        self.assertEqual(integrated_image.shape, self.image.shape)
        np.testing.assert_array_equal(integrated_image, self.image)

if __name__ == '__main__':
    unittest.main()