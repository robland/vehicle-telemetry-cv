import pytest
import cv2


from vision.test_pipeline import FakeVideoCapture, FakeDetector

from monitoring_taxi.vision.benchmark import benchmark_frame_skip
from monitoring_taxi.vision.pipeline import VisionPipeline

from vision.test_pipeline import FakeTracker


def test_benchmark_valid_frame_skip(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    fake_tracker = FakeTracker(max_distance=0.1)

    metrics = benchmark_frame_skip(
        VisionPipeline,
        fake_detector,
        fake_tracker,
        "fake_video.mp4",
        frame_skips=[0, 1, 2]
    )
    assert len(metrics) == 3
    for metric in metrics:
        print(metric)
        assert metric.frames_read == 3
        assert metric.frames_processed == 3

        assert metric.frames_read == 3
        assert metric.frames_processed == 2

        assert metric.frames_read == 3
        assert metric.frames_processed == 1
        assert fake_detector.calls == 6
