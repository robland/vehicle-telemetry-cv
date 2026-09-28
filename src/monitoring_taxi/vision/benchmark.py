from dataclasses import asdict
from ultralytics import YOLO

from monitoring_taxi.vision.metrics import PipelineMetrics
from monitoring_taxi.vision.pipeline import VisionPipeline
from monitoring_taxi.vision.detector import PersonDetector
from monitoring_taxi.vision.tracker import SimpleTracker

from src.monitoring_taxi.vision.tracker import GlobalTracker


def benchmark_frame_skip(pipeline_factory, detector: PersonDetector, tracker: SimpleTracker, source: str,
                         frame_skips: list[int]):
    metrics = []
    for frame_skip in frame_skips:
        pipeline = pipeline_factory(
            detector=detector,
            tracker=tracker,
            target_size=(640, 640),
            frame_skip=frame_skip
        )
        metrics.append(
                pipeline.run(source)
        )
    return metrics


def benchmark_pipeline_with_distance():
    frame_skips = [0]
    distances = [0.10, 0.125, 0.15, 0.175, 0.20]
    for distance in distances:
        for frame_skip in frame_skips:
            tracker = SimpleTracker(max_distance=distance)
            detector = PersonDetector(
                model=YOLO("yolo11n.pt"),
                confidence_threshold=0.5,
                device="cpu"
            )
            results = benchmark_frame_skip(
                VisionPipeline,
                detector,
                tracker,
                "data/videos/taxi_v0.mp4",
                frame_skips=[frame_skip]
            )
            metrics = results[0]

            print({
                "frame_skip": frame_skip,
                "frames_read": metrics.frames_read,
                "frames_processed": metrics.frames_processed,
                "elapsed_time": metrics.elapsed_time,
                "inference_time_total": metrics.inference_time_total,
                "total_detections": metrics.total_detections,
                "tracks_created": metrics.tracks_created,
                "average_track_length": metrics.average_track_length,
                "median_track_length": metrics.median_track_length,
                "short_track_ratio": metrics.short_track_ratio
            })


def benchmark_pipeline_with_global_tracker():
    frame_skips = [0]
    distance = 0.15
    trackers = [
        SimpleTracker(max_distance=distance),
        GlobalTracker(max_missed_frames=10, alpha=0.5, association_threshold=0.5)
    ]

    for tracker in trackers:
        for frame_skip in frame_skips:
            detector = PersonDetector(
                model=YOLO("yolo11n.pt"),
                confidence_threshold=0.5,
                device="cpu"
            )
            results = benchmark_frame_skip(
                VisionPipeline,
                detector,
                tracker,
                "data/videos/taxi_v0.mp4",
                frame_skips=[frame_skip]
            )
            metrics = results[0]

            print({
                "frame_skip": frame_skip,
                "frames_read": metrics.frames_read,
                "frames_processed": metrics.frames_processed,
                "elapsed_time": metrics.elapsed_time,
                "inference_time_total": metrics.inference_time_total,
                "total_detections": metrics.total_detections,
                "tracks_created": metrics.tracks_created,
                "average_track_length": metrics.average_track_length,
                "median_track_length": metrics.median_track_length,
                "short_track_ratio": metrics.short_track_ratio
            })


if __name__ == '__main__':
    benchmark_pipeline_with_global_tracker()
