from typing import Any

import numpy as np
from PIL import Image

from monitoring_taxi.vision.bbox import bbox_to_original, normalize_bbox
from monitoring_taxi.vision.detection import PersonDetection


class PersonDetector:
    def __init__(self, model=None, confidence_threshold: float = 0.5, device: str = "cpu"):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"Expected confidence_threshold between [0, 1].")
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, frame: Image.Image | np.ndarray) -> list[Any]:
        return self.model.predict(frame, imgsz=640, conf=self.confidence_threshold, verbose=False)

    def postprocess(self, raw_results, transform) -> list[PersonDetection]:
        return self._convert_results(raw_results, transform)

    def _convert_results(self, detections, transform: dict[str, Any]) -> list[PersonDetection]:
        person_detections = []
        for result in detections:
            for index, xyxy in enumerate(result.boxes.xyxy):
                class_id = int(result.boxes.cls[index])

                if class_id != 0:
                    continue

                class_name = "person"

                bbox = {
                    "x_min": float(xyxy[0]),
                    "y_min": float(xyxy[1]),
                    "x_max": float(xyxy[2]),
                    "y_max": float(xyxy[3]),
                }

                original_bbox = bbox_to_original(bbox, transform)
                normalized_bbox = normalize_bbox(
                    original_bbox,
                    transform["original_width"],
                    transform["original_height"]
                )

                person = PersonDetection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(result.boxes.conf[index]),
                    bbox=normalized_bbox
                )

                person_detections.append(person)

        return person_detections



