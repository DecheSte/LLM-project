import math
from dataclasses import dataclass


def posture_distance(points, confidence=0.5, normalized=False):
    """鼻子到双肩中点的有符号垂直距离；无效关键点返回 None。"""
    if len(points) < 7:
        return None
    nose, left, right = (points[i] for i in (0, 5, 6))
    if any(len(p) < 3 or not all(math.isfinite(float(v)) for v in p[:3])
           or p[2] < confidence for p in (nose, left, right)):
        return None
    distance = (left[1] + right[1]) / 2 - nose[1]
    if normalized:
        width = math.hypot(left[0] - right[0], left[1] - right[1])
        if width < 1:
            return None
        distance /= width
    return float(distance)


@dataclass
class PostureTimer:
    threshold: float
    duration: float
    cooldown: float
    max_gap: float = 1.5
    start: float | None = None
    last: float | None = None
    eligible_at: float = 0.0

    def update(self, distance, now):
        if self.last is not None and now - self.last > self.max_gap:
            self.start = None
        self.last = now
        if distance is None or distance >= self.threshold or now < self.eligible_at:
            self.start = None
            return False, 0.0
        if self.start is None:
            self.start = now
        elapsed = now - self.start
        return elapsed >= self.duration, elapsed

    def reset(self, now, delay=None):
        self.start = self.last = None
        self.eligible_at = now + (self.cooldown if delay is None else delay)
