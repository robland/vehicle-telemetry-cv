import cv2
from PIL import Image
import time

from monitoring_taxi.vision.metrics import PipelineMetrics
from monitoring_taxi.vision.preprocessing import letterbox
from monitoring_taxi.vision.detector import PersonDetector
from monitoring_taxi.vision.tracker import SimpleTracker


class VisionPipeline:
    def __init__(self, detector: PersonDetector, tracker: SimpleTracker, target_size: tuple[int, int] = (640, 640), frame_skip: int = 0):
        self.detector = detector
        self.tracker = tracker
        self.target_size = target_size

        if frame_skip < 0:
            raise ValueError(f"Expected frame_skip >= 0 but got {frame_skip}")

        self.frame_skip = frame_skip

    def run(self, source: str, frame_callback=None) -> PipelineMetrics:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}.")

        start_total = time.perf_counter()
        frames_processed = 0
        frames_read = 0
        preprocessing_time_total = 0.0
        postprocessing_time_total = 0.0
        detector_time_total = 0.0
        total_detections = 0

        try:
            frame_index = 0

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frames_read += 1

                if frame_index % (self.frame_skip + 1) != 0:
                    frame_index += 1
                    continue
                # -------------------------
                # Preprocessing
                # -------------------------
                start_preprocessing = time.perf_counter()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame_rgb)

                padded_img, transform = letterbox(
                    frame_image,
                    target_width=self.target_size[0],
                    target_height=self.target_size[1]
                )

                preprocessing_time_total += time.perf_counter() - start_preprocessing

                # -------------------------
                # Inference
                # -------------------------
                start_detector = time.perf_counter()
                results = self.detector.detect(padded_img)
                detector_time_total += time.perf_counter() - start_detector

                # -------------------------
                # Postprocessing
                # -------------------------
                start_postprocessing = time.perf_counter()
                persons = self.detector.postprocess(results, transform)

                tracks = self.tracker.update(persons)

                # Draw rectangle for detection
                if frame_callback is not None:
                    frame_callback(frame, persons, tracks)

                total_detections += len(persons)
                postprocessing_time_total += time.perf_counter()-start_postprocessing

                frames_processed += 1
                frame_index += 1
        finally:
            cap.release()
        elapsed = time.perf_counter() - start_total
        return PipelineMetrics(
            frames_read=frames_read,
            frames_processed=frames_processed,
            elapsed_time=elapsed,
            preprocessing_time_total=preprocessing_time_total,
            inference_time_total=detector_time_total,
            postprocessing_time_total=postprocessing_time_total,
            total_detections=total_detections,
            tracks_created=self.tracker.tracks_created,
            track_frames=self.tracker.track_frames
        )


if __name__ == "__main__":
    from ultralytics import YOLO

    skip = [0, 1, 2, 3, 4]

    detector = PersonDetector(
        model=YOLO("yolo11n.pt"),
        confidence_threshold=0.5,
        device="cpu"
    )

    for s in skip:
        pipeline = VisionPipeline(
            detector=detector,
            target_size=(640, 640),
            frame_skip=s
        )

        metrics = pipeline.run(
            "data/videos/taxi_v0.mp4"
        )

        print(s, metrics)
