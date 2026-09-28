import json

import pytest
import numpy as np
import cv2

from monitoring_taxi.vision.detection import PersonDetection
from monitoring_taxi.vision.visualization import draw_tracks
from monitoring_taxi.vision.tracker import Track

from vision.test_pipeline import FakeVideoCapture

from monitoring_taxi.vision.visualization import generate_tracking_video


def putText(annotated_frame: np.ndarray, text: str, xy, font:int, fontScale:int, color:tuple, thickness:int,
            lineType:int, bottomLeftOrigin):
    return text


def test_frame_without_track():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_index = 0
    tracks = []

    frame_processed = draw_tracks(frame_index, frame, tracks)

    assert np.array_equal(frame_processed, frame)


def test_one_frame_with_one_track(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_index = 0
    detection_1 = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )
    track = Track(track_id=1, detection=detection_1)
    tracks = [track, ]

    frame_processed = draw_tracks(frame_index, frame, tracks)
    assert not np.array_equal(frame_processed, frame)
    assert np.any(frame_processed[216, 384] != 0)


def test_track_id_is_drawn(monkeypatch):

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame_index = 0
    detection = PersonDetection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox={
            "x_min": 0.2,
            "y_min": 0.2,
            "x_max": 0.4,
            "y_max": 0.6,
        },
    )

    track = Track(
        track_id=42,
        detection=detection,
    )

    captured = {}

    def fake_put_text(
            image,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
            line_type=None,
            bottom_left_origin=None,
    ):
        captured["text"] = text
        captured["position"] = position

        return image

    monkeypatch.setattr(
        cv2,
        "putText",
        fake_put_text,
    )

    draw_tracks(frame_index, frame, [track])

    assert captured["text"] == "ID: 42"


class FakeVideoWriter:

    def __init__(self, output, fourcc, fps, imgsz):
        self.output = output
        self.fourcc = fourcc
        self.fps = fps
        self.imgsz = imgsz
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True

    def get(self, fps):
        self.fps = fps


class FakePipeline:

    def __init__(self, detector=None, tracker=None, target_size=None, frame_skip=0):
        self.frame_skip = frame_skip
        self.callback = None

    def run(self, source, frame_callback=None):
        self.callback = frame_callback

        frame = np.zeros(
            (1080, 1920, 3),
            dtype=np.uint8,
        )

        persons = []

        tracks = []

        if frame_callback is not None:
            frame_callback(frame, persons, tracks)


def test_generate_tracking_video(monkeypatch):
    created_writers = []

    def fake_video_writer(output, fourcc, fps, imgsz):
        writer = FakeVideoWriter(
            output,
            fourcc,
            fps,
            imgsz,
        )

        created_writers.append(writer)

        return writer

    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        fake_video_writer
    )

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture,
    )

    pipeline = FakePipeline()

    generate_tracking_video(
        source="fake_video.mp4",
        output="tracking_output.mp4",
        pipeline=pipeline,
    )

    assert len(created_writers) == 1

    writer = created_writers[0]

    assert writer.output == "tracking_output.mp4"
    assert writer.fps == 30.0
    assert writer.imgsz == (1920, 1080)

    assert len(writer.frames) == 1
    assert writer.released is True


def test_generate_tracking_video_with_tracking_file(monkeypatch, tmp_path):
    created_writers = []

    def fake_video_writer(output, fourcc, fps, imgsz):
        writer = FakeVideoWriter(
            output,
            fourcc,
            fps,
            imgsz,
        )

        created_writers.append(writer)

        return writer

    monkeypatch.setattr(
        cv2,
        "VideoWriter",
        fake_video_writer
    )

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        FakeVideoCapture,
    )

    pipeline = FakePipeline()

    tracking_output = tmp_path / "fake_v0_tracking_output.jsonl"

    generate_tracking_video(
        source="fake_video.mp4",
        output="tracking_output.mp4",
        pipeline=pipeline,
        tracking_output=tracking_output
    )

    assert tracking_output.exists()

    with tracking_output.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
        ]

    assert len(records) == 1

    assert records[0]["frame_index"] == 0

    for record in records:
        assert "tracks" in record

