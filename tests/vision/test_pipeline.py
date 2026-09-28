from typing import Any

import cv2
import numpy as np
from PIL.Image import Image
import pytest


from monitoring_taxi.vision.metrics import PipelineMetrics
from monitoring_taxi.vision.pipeline import VisionPipeline
from monitoring_taxi.vision.tracker import Track
from monitoring_taxi.vision.tracker import SimpleTracker

from monitoring_taxi.vision.detection import PersonDetection


def test_pipeline_valid_metrics():
    metrics = PipelineMetrics(
        frames_read=1,
        frames_processed=1,
        elapsed_time=30,
        preprocessing_time_total=5,
        inference_time_total=10,
        postprocessing_time_total=15,
        total_detections=1,
        tracks_created=1,
        track_frames={}
    )
    assert metrics.processing_fps == pytest.approx(1/30, rel=1e-3)
    assert metrics.average_preprocessing_ms == pytest.approx(5/1*1000, rel=1e-3)
    assert metrics.average_inference_ms == pytest.approx(10/1*1000, rel=1e-3)
    assert metrics.average_postprocessing_ms == pytest.approx(15 * 1000)


def test_pipeline_valid_processing_ratio():
    metrics = PipelineMetrics(
        frames_read=5,
        frames_processed=1,
        elapsed_time=30,
        preprocessing_time_total=5,
        inference_time_total=10,
        postprocessing_time_total=15,
        total_detections=1,
        tracks_created=1,
        track_frames={}
    )
    assert metrics.processing_ratio == pytest.approx(1/5, rel=1e-3)


def test_pipeline_invalid_negative_metrics():
    with (pytest.raises(ValueError) as e):
        PipelineMetrics(
            frames_read=1,
            frames_processed=1,
            elapsed_time=-30,
            preprocessing_time_total=5,
            inference_time_total=10,
            postprocessing_time_total=15,
            total_detections=1,
            tracks_created=1,
            track_frames={}
        )
    assert str(e.value) == "Expected elapsed_time > 0."


def test_pipeline_invalid_frame_processed_metrics():
    with (pytest.raises(ValueError) as e):
        PipelineMetrics(
            frames_read=1,
            frames_processed=0,
            elapsed_time=30,
            preprocessing_time_total=10,
            inference_time_total=10,
            postprocessing_time_total=15,
            total_detections=1,
            tracks_created=1,
            track_frames={}
        )
    assert str(e.value) == "Expected frames_processed >= 0."


def test_pipeline_invalid_preprocessing_time_total_metrics():
    with (pytest.raises(ValueError) as e):
        PipelineMetrics(
            frames_read=1,
            frames_processed=1,
            elapsed_time=30,
            preprocessing_time_total=-10,
            inference_time_total=10,
            postprocessing_time_total=15,
            total_detections=1,
            tracks_created=1,
            track_frames={}
        )
    assert str(e.value) == "Expected preprocessing_time_total >= 0."


def test_pipeline_invalid_postprocessing_time_total_metrics():
    with (pytest.raises(ValueError) as e):
        PipelineMetrics(
            frames_read=1,
            frames_processed=1,
            elapsed_time=30,
            preprocessing_time_total=10,
            inference_time_total=10,
            postprocessing_time_total=-15,
            total_detections=1,
            tracks_created=1,
            track_frames={}
        )
    assert str(e.value) == "Expected postprocessing_time_total >= 0."


def test_pipeline_invalid_inference_time_total_metrics():
    with (pytest.raises(ValueError) as e):
        PipelineMetrics(
            frames_read=1,
            frames_processed=1,
            elapsed_time=30,
            preprocessing_time_total=10,
            inference_time_total=-10,
            postprocessing_time_total=-15,
            total_detections=1,
            tracks_created=1,
            track_frames={}
        )
    assert str(e.value) == "Expected inference_time_total >= 0."


class FakeResult:
    pass


class FakePersonDetection:

    def __init__(self):
        self.bbox = {
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.2,
            "y_max": 0.6,
        }


class FakeTracker:
    def __init__(self, max_distance=0.1):
        self.max_distance = max_distance
        self.calls = 0
        self.last_detections = None
        self.last_tracks = None
        self.tracks_created = 0
        self.track_frames = {}
        self.track = Track(
            track_id=0,
            detection=FakePersonDetection()

        )

    def update(self, detection: list[FakePersonDetection]) -> list[Track]:
        self.calls += 1
        self.last_tracks = [self.track, ]
        return self.last_tracks


class FakeDetector:
    def __init__(self, detections_per_frame=None):
        self.calls = 0
        self.detections_per_frame = detections_per_frame \
            if detections_per_frame is not None else \
            [[FakePersonDetection()], [FakePersonDetection()], [FakePersonDetection()]]

    def detect(self, frame: Image):
        self.calls += 1
        results = []
        return results

    def postprocess(self, raw_results, transform):
        index = (self.calls - 1) % len(self.detections_per_frame)
        return self.detections_per_frame[index]


class FakeVideoCapture:
    def __init__(self, source: str, frames=None):
        if frames is None:
            frames = [
                np.zeros((1080, 1920, 3), dtype=np.uint8),
                np.zeros((1080, 1920, 3), dtype=np.uint8),
                np.zeros((1080, 1920, 3), dtype=np.uint8),
            ]

        self.frames = frames
        self.index = 0

    def read(self):

        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def isOpened(self):
        return True

    def release(self):
        pass

    def get(self, property_id):

        if property_id == cv2.CAP_PROP_FPS:
            return 30.0

        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 1920

        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 1080

        return 0.0


def test_pipeline_run_pipeline_without_video(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    tracker = FakeTracker()
    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=tracker,
        target_size=(640,  640)
    )
    metrics = pipeline.run("fake_video.mp4")
    assert metrics.frames_processed == 3
    assert fake_detector.calls == 3


def test_pipeline_frame_skip(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    fake_tracker = FakeTracker()
    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=fake_tracker,
        target_size=(640,  640),
        frame_skip=1
    )
    metrics = pipeline.run("fake_video.mp4")
    assert metrics.frames_read == 3
    assert metrics.frames_processed == 2
    assert fake_detector.calls == 2


def test_pipeline_valid_total_detections(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    fake_tracker = FakeTracker()
    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=fake_tracker,
        target_size=(640,  640),
        frame_skip=0
    )
    metrics = pipeline.run("fake_video.mp4")
    assert metrics.frames_read == 3
    assert metrics.frames_processed == 3
    assert metrics.total_detections == 3
    assert fake_detector.calls == 3


def test_pipeline_frame_skip_total_detections(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    fake_tracker = FakeTracker()
    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=fake_tracker,
        target_size=(640, 640),
        frame_skip=1
    )

    metrics = pipeline.run("fake_video.mp4")

    assert metrics.frames_read == 3
    assert metrics.frames_processed == 2
    assert metrics.total_detections == 2
    assert fake_detector.calls == 2


def test_pipeline_frame_skip_2_total_detections(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    fake_tracker = FakeTracker()
    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=fake_tracker,
        target_size=(640, 640),
        frame_skip=2
    )

    metrics = pipeline.run("fake_video.mp4")

    assert metrics.frames_read == 3
    assert metrics.frames_processed == 1
    assert metrics.total_detections == 1
    assert fake_detector.calls == 1
    assert metrics.average_detections_per_processed_frame == 1.0


def fake_video_capture_one_frame(source):
    return FakeVideoCapture(
        source,
        frames=[
            np.zeros((1080, 1920, 3), dtype=np.uint8),
        ],
    )


def test_pipeline_good_track_generation(monkeypatch):

    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.2,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.7,
            "y_min": 0.2,
            "x_max": 0.8,
            "y_max": 0.6,
        },
    )
    fake_detector = FakeDetector(detections_per_frame=[[detection_1, detection_2]])
    tracker = SimpleTracker()

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        fake_video_capture_one_frame
    )

    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=tracker,
        target_size=(640, 640),
        frame_skip=0
    )
    metrics = pipeline.run("fake_video.mp4")
    assert metrics.total_detections == 2
    assert len(tracker.tracks) == 2
    assert tracker.tracks[1].track_id == 2


def test_pipeline_calls_tracker(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )
    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.1,
            "y_min": 0.2,
            "x_max": 0.2,
            "y_max": 0.6,
        },
    )

    detection_2 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.7,
            "y_min": 0.2,
            "x_max": 0.8,
            "y_max": 0.6,
        },
    )

    detections_per_frame = [[detection_1, detection_2], [detection_1, detection_2], [detection_1, detection_2]]
    fake_detector = FakeDetector(detections_per_frame=detections_per_frame)
    fake_tracker = FakeTracker()

    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=fake_tracker,
        target_size=(640, 640),
        frame_skip=0
    )

    metrics = pipeline.run("fake_video.mp4")
    assert metrics.frames_processed == 3
    assert fake_detector.calls == 3
    assert fake_tracker.calls == 3


def test_pipeline_keeps_track_id(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )
    detection_1 = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "x_min": 0.5,
            "y_min": 0.1,
            "x_max": 0.6,
            "y_max": 0.8,
        }
    }

    detection_2 = {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "bbox": {
            "x_min": 0.5,
            "y_min": 0.1,
            "x_max": 1,
            "y_max": 1,
        }
    }
    detections_per_frame = [
        [PersonDetection(**detection_1)],
        [PersonDetection(**detection_1), PersonDetection(**detection_2)],
        [PersonDetection(**detection_1), PersonDetection(**detection_2)]
    ]

    fake_detector = FakeDetector(detections_per_frame=detections_per_frame)
    tracker = SimpleTracker()

    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=tracker,
        target_size=(640, 640),
        frame_skip=0
    )

    metrics = pipeline.run("fake_video.mp4")
    assert metrics.frames_processed == 3
    assert fake_detector.calls == 3
    assert len(tracker.tracks) == 2
    assert tracker.tracks[0].track_id == 1
    assert tracker.tracks[1].track_id == 2
    assert tracker.track_frames[1] == 3


def test_pipeline_track_frames(monkeypatch):
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture
    )

    fake_detector = FakeDetector()
    tracker = SimpleTracker(max_distance=0.1)

    pipeline = VisionPipeline(
        detector=fake_detector,
        tracker=tracker,
        target_size=(640, 640),
        frame_skip=0
    )

    metrics = pipeline.run("fake_video.mp4")

    assert metrics.track_frames == tracker.track_frames


def test_pipeline_calls_frame_callback(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)

    detector = FakeDetector()
    tracker = FakeTracker(max_distance=0.1)

    callback_calls = []

    def callback(frame, persons, tracks):
        callback_calls.append(
            {
                "frame": frame,
                "tracks": tracks,
                "persons": persons
            }
        )

    pipeline = VisionPipeline(
        detector=detector,
        tracker=tracker,
    )

    pipeline.run(
        "fake_video.mp4",
        frame_callback=callback,
    )

    assert len(callback_calls) == 3
    assert len(callback_calls[0]["tracks"]) > 0
