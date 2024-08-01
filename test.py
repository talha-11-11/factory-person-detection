import unittest
from scripts.person_detection import detect_persons

class TestPersonDetection(unittest.TestCase):
    def test_detect_persons(self):
        detect_persons('data/input_videos/video1.mp4', 'data/output/detected_frames/')
        self.assertTrue(os.path.exists('data/output/detected_frames/frame_00000.jpg'))

if __name__ == "__main__":
    unittest.main()
