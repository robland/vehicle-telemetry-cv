from datetime import datetime

from ultralytics import YOLO

from monitoring_taxi.vision.detector import PersonDetector
from monitoring_taxi.vision.tracker import SimpleTracker, GlobalTracker
from monitoring_taxi.vision.pipeline import VisionPipeline
from monitoring_taxi.vision.visualization import generate_tracking_video


str_current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
VIDEO_SOURCE = "data/videos/taxi_v0.mp4"
VIDEO_OUTPUT = f"data/videos/tracking_output_v0_{str_current_datetime}.mp4"
TRACKING_OUTPUT = f"data/tracking/tracking_output_v0_{str_current_datetime}.jsonl"


def main():
    detector = PersonDetector(
        model=YOLO("yolo11n.pt"),
        confidence_threshold=0.5,
        device="cpu"
    )

    # tracker = SimpleTracker(
    #    max_distance=0.15,
    #   max_missed_frames=1
    # )

    tracker = GlobalTracker(max_missed_frames=1)
    pipeline = VisionPipeline(
        detector=detector,
        tracker=tracker,
        target_size=(640, 640),
        frame_skip=0
    )

    return generate_tracking_video(
        source=VIDEO_SOURCE,
        output=VIDEO_OUTPUT,
        pipeline=pipeline,
        tracking_output=TRACKING_OUTPUT
    )


if __name__ == "__main__":
    metrics = main()
