from dataclasses import dataclass

from monitoring_taxi.vision.detection import PersonDetection
from monitoring_taxi.vision.tracking_utils import (hungarian_assignment,
                                                   build_cost_matrix,
                                                   filter_matches)


@dataclass
class Track:
    track_id: int
    detection: PersonDetection
    frames_since_update: int = 0


class SimpleTracker:
    def __init__(self, max_distance: float = 0.1, max_missed_frames: int = 2):
        if max_distance <= 0:
            raise ValueError(
                f"Expected max_distance > 0 but got {max_distance}."
            )
        if max_missed_frames < 0:
            raise ValueError(
                f"Expected max_missed_frames >= 0 but got {max_missed_frames}."
            )
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_track_id = 1
        self.tracks_created = 0
        self.tracks: list[Track] = []
        self.track_frames: dict[int, int] = {}

    def update(self, detections: list[PersonDetection]) -> list[Track]:
        if not detections:
            active_tracks = []

            for track in self.tracks:
                track.frames_since_update += 1

                if track.frames_since_update <= self.max_missed_frames:
                    active_tracks.append(track)

            self.tracks = active_tracks

            return active_tracks

        new_tracks = []
        used_track_ids = set()

        for detection in detections:
            center_x = (detection.bbox["x_min"] + detection.bbox["x_max"])/2
            center_y = (detection.bbox["y_min"] + detection.bbox["y_max"])/2

            best_track = None
            best_distance = float("inf")

            for track in self.tracks:
                if track.track_id in used_track_ids:
                    continue
                previous_detection = track.detection

                prev_center_x = (previous_detection.bbox["x_min"] + previous_detection.bbox["x_max"])/2
                prev_center_y = (previous_detection.bbox["y_min"] + previous_detection.bbox["y_max"])/2

                distance = ((center_x-prev_center_x)**2 + (center_y-prev_center_y)**2) ** 0.5

                if best_distance > distance <= self.max_distance:
                    best_distance = distance
                    best_track = track

            if best_track is None:
                track = Track(
                    track_id=self.next_track_id,
                    detection=detection,
                    frames_since_update=0
                )
                self.next_track_id += 1
                self.tracks_created += 1

            else:
                track = Track(
                    track_id=best_track.track_id,
                    detection=detection,
                    frames_since_update=0
                )
                used_track_ids.add(best_track.track_id)

            new_tracks.append(track)
            track_id = track.track_id
            self.track_frames[track_id] = (
                self.track_frames.get(track_id, 0) + 1
            )
        self.tracks = new_tracks
        return new_tracks


class GlobalTracker:
    def __init__(self, max_missed_frames: int = 1, alpha: float = 0.5, association_threshold: float = 0.5):

        if max_missed_frames < 0:
            raise ValueError(
                f"Expected max_missed_frames >= 0 "
                f"but got {max_missed_frames}."
            )

        if not 0 <= alpha <= 1:
            raise ValueError(
                f"Expected alpha between [0, 1] "
                f"but got {alpha}."
            )

        if association_threshold < 0:
            raise ValueError(
                f"Expected association_threshold >= 0 "
                f"but got {association_threshold}."
            )

        self.max_missed_frames = max_missed_frames
        self.alpha = alpha
        self.association_threshold = association_threshold

        self.next_track_id = 1
        self.tracks_created = 0
        self.track_frames: dict[int, int] = {}
        self.tracks: list[Track] = []

    def update(self, detections: list[PersonDetection]) -> list[Track]:
        if not self.tracks:
            return self._create_tracks(detections)

        cost_matrix = build_cost_matrix(
            self.tracks,
            detections,
            alpha=self.alpha
        )

        matches = hungarian_assignment(cost_matrix)
        accepted_matches, unmatched_track_indices, unmatched_detection_indices = filter_matches(
            matches,
            cost_matrix,
            association_threshold=self.association_threshold
        )

        new_tracks = []

        for track_index, detection_index in accepted_matches:
            previous_track = self.tracks[track_index]

            track = Track(
                track_id=previous_track.track_id,
                detection=detections[detection_index],
                frames_since_update=0,
            )

            self.track_frames[track.track_id] = (
                self.track_frames.get(track.track_id, 0) + 1
            )

            new_tracks.append(track)

        for track_index in unmatched_track_indices:
            previous_track = self.tracks[track_index]

            frame_since_update = (
                previous_track.frames_since_update + 1
            )

            if frame_since_update <= self.max_missed_frames:
                track = Track(
                    track_id=previous_track.track_id,
                    detection=previous_track.detection,
                    frames_since_update=frame_since_update
                )

                self.track_frames[track.track_id] = self.track_frames.get(track.track_id, 0) + 1

                new_tracks.append(track)

        for detection_index in unmatched_detection_indices:
            track = Track(
                track_id=self.next_track_id,
                detection=detections[detection_index],
                frames_since_update=0
            )

            self.next_track_id += 1
            self.tracks_created += 1
            self.track_frames[track.track_id] = 1
            new_tracks.append(track)

        self.tracks = new_tracks
        return new_tracks

    def _create_tracks(self, detections: list[PersonDetection]) -> list[Track]:
        tracks = []

        for detection in detections:
            track = Track(
                track_id=self.next_track_id,
                detection=detection,
                frames_since_update=0
            )

            self.next_track_id += 1
            self.tracks_created += 1
            self.track_frames[track.track_id] = 1

            tracks.append(track)

        self.tracks = tracks

        return tracks
