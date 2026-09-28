import json
from dataclasses import dataclass, asdict

import numpy as np
import cv2
from ultralytics import YOLO

from monitoring_taxi.vision.tracker import Track, SimpleTracker
from monitoring_taxi.vision.pipeline import VisionPipeline
from monitoring_taxi.vision.detector import PersonDetector

from monitoring_taxi.vision.metrics import PipelineMetrics


def draw_tracks(frame_index: int, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    annotated_frame = frame.copy()
    height, width = annotated_frame.shape[:2]

    for track in tracks:
        bbox = track.detection.bbox

        x_min = int(bbox["x_min"] * width)
        y_min = int(bbox["y_min"] * height)
        x_max = int(bbox["x_max"] * width)
        y_max = int(bbox["y_max"] * height)

        cv2.rectangle(
            annotated_frame,
            (x_min, y_min),
            (x_max, y_max),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_index}",
            (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"ID: {track.track_id}",
            (x_min, max(y_min - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )
    return annotated_frame


def generate_tracking_video(source: str, output: str, pipeline: VisionPipeline, tracking_output: str | None = None) -> (
        PipelineMetrics | None):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        fps = 30.0

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video dimensions: {width}x{height}."
        )

    if pipeline.frame_skip != 0:
        raise ValueError(
            "generate_tracking_video currently requires frame_skip=0."
        )

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise ValueError(
            f"Could not open video writer: {output}."
        )

    tracking_file = None

    if tracking_output is not None:
        tracking_file = open(
            tracking_output,
            "w",
            encoding="utf-8"
        )

    frame_index = 0

    def callback(frame, persons, tracks):
        nonlocal frame_index
        annotated_frame = draw_tracks(frame_index, frame, tracks)
        writer.write(annotated_frame)

        if tracking_file is not None:
            record = {
                "frame_index": frame_index,
                "detections": [asdict(person) for person in persons],
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "bbox": track.detection.bbox,
                        "confidence": track.detection.confidence,
                    }
                    for track in tracks
                ],
            }

            tracking_file.write(
                json.dumps(record) + "\n"
            )
        frame_index += 1

    try:
        return pipeline.run(
            source,
            frame_callback=callback,
        )
    finally:
        writer.release()

        if tracking_file is not None:
            tracking_file.close()


if __name__ == "__main__":
    source = "data/videos/taxi_v0.mp4"
    output = "data/videos/annotated_taxi_v0.mp4"

    detector = PersonDetector(
        model=YOLO("yolo11n.pt"),
        confidence_threshold=0.5,
        device="cpu"
    )
    tracker = SimpleTracker(max_missed_frames=30)
    pipeline = VisionPipeline(
        detector=detector,
        tracker=SimpleTracker(max_distance=0.15),
        target_size=(640, 640),
        frame_skip=0
    )

    metrics = generate_tracking_video(source, output, pipeline)
    print(metrics)
