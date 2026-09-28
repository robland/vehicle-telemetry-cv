from dataclasses import dataclass

import numpy as np


@dataclass
class PipelineMetrics:
    frames_read: int
    frames_processed: int
    elapsed_time: float
    preprocessing_time_total: float
    inference_time_total: float
    postprocessing_time_total: float
    total_detections: int
    tracks_created: int
    track_frames: dict[int, int]

    def __post_init__(self):
        if self.elapsed_time <= 0:
            raise ValueError("Expected elapsed_time > 0.")

        if self.frames_processed <= 0:
            raise ValueError("Expected frames_processed >= 0.")

        if self.frames_read <= 0:
            raise ValueError("Expected frames_read >= 0.")

        if self.preprocessing_time_total < 0:
            raise ValueError("Expected preprocessing_time_total >= 0.")

        if self.inference_time_total < 0:
            raise ValueError("Expected inference_time_total >= 0.")

        if self.postprocessing_time_total < 0:
            raise ValueError("Expected postprocessing_time_total >= 0.")
        if self.total_detections < 0:
            raise ValueError("Expected detections >= 0.")

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.elapsed_time

    @property
    def input_fps(self) -> float:
        return self.frames_read / self.elapsed_time

    @property
    def average_preprocessing_ms(self) -> float:
        return self.preprocessing_time_total / self.frames_processed * 1000

    @property
    def average_inference_ms(self) -> float:
        return (self.inference_time_total / self.frames_processed) * 1000

    @property
    def average_postprocessing_ms(self) -> float:
        return self.postprocessing_time_total / self.frames_processed * 1000

    @property
    def processing_ratio(self) -> float:
        return self.frames_processed / self.frames_read

    @property
    def average_detections_per_processed_frame(self) -> float:
        return self.total_detections / self.frames_processed

    @property
    def average_track_length(self) -> float:
        if not self.track_frames:
            return 0.0
        return sum(self.track_frames.values()) / len(self.track_frames)

    @property
    def median_track_length(self) -> float:
        if not self.track_frames:
            return 0.0

        lengths = list(self.track_frames.values())
        return float(np.median(lengths))

    @property
    def short_track_ratio(self) -> float:
        if not self.track_frames:
            return 0.0

        short_tracks = sum(
            1 for length in self.track_frames.values()
            if length <= 2
        )

        return short_tracks / len(self.track_frames)