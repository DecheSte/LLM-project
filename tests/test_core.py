import unittest
from posture_robot.core import PostureTimer, posture_distance


class CoreTests(unittest.TestCase):
    def test_continuous_time_and_threshold(self):
        timer = PostureTimer(120, 5, 30)
        for t in range(5):
            self.assertFalse(timer.update(100, t)[0])
        self.assertTrue(timer.update(100, 5)[0])
        self.assertEqual(timer.update(120, 6), (False, 0))

    def test_missing_and_gap_reset(self):
        timer = PostureTimer(120, 2, 30)
        timer.update(100, 0)
        timer.update(100, 1)
        timer.update(None, 1.1)
        self.assertEqual(timer.update(100, 2), (False, 0))
        self.assertEqual(timer.update(100, 10), (False, 0))

    def test_confirmation_cooldown_requires_new_full_interval(self):
        timer = PostureTimer(120, 2, 30)
        timer.reset(10)
        self.assertFalse(timer.update(100, 39)[0])
        self.assertEqual(timer.update(100, 40), (False, 0))
        timer.update(100, 41)
        self.assertTrue(timer.update(100, 42)[0])

    def test_keypoint_confidence_and_normalization(self):
        points = [[0, 0, 1] for _ in range(17)]
        points[0], points[5], points[6] = [100, 50, 1], [0, 150, 1], [200, 150, 1]
        self.assertEqual(posture_distance(points), 100)
        self.assertEqual(posture_distance(points, normalized=True), 0.5)
        points[0][2] = 0.1
        self.assertIsNone(posture_distance(points))
        points[0][2] = float("nan")
        self.assertIsNone(posture_distance(points))


if __name__ == "__main__":
    unittest.main()
