from collections import Counter, defaultdict
import json
import numpy as np


def calculate_track_length_stats(track_frames: dict[int, int]) -> dict:
    lengths = list(track_frames.values())

    if not lengths:
        return {
            "count": 0
        }

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75, method='nearest')),
        "p90": float(np.percentile(lengths, 90)),
        "track_ge_10": sum(length >= 10 for length in lengths),
        "track_ge_30": sum(length >= 30 for length in lengths),
        "track_ge_100": sum(length >= 100 for length in lengths),
        "track_ge_500": sum(length >= 500 for length in lengths),
    }


def analyze_tracking_file(source: str):
    frames = []
    track_frames = Counter()

    with open(source, "r", encoding="utf-8") as tracking_file:

        for line in tracking_file:
            data = json.loads(line)

            frames.append(data)

            for track in data["tracks"]:
                track_frames[track["track_id"]] += 1

    detection_counts = [
        len(frame["detections"]) for frame in frames
    ]

    track_counts = [
        len(frame["tracks"]) for frame in frames
    ]

    track_lifetimes = analyze_track_lifetimes(frames)
    return {
        "frames": {
            "total": len(frames),
            "with_detections": sum(
                count > 0 for count in detection_counts
            ),
            "without_detections": sum(
                count == 0 for count in detection_counts
            )
        },
        "detections": {
            "total": sum(detection_counts),
            "mean_per_frame": float(np.mean(detection_counts)),
            "distributions": dict(
                sorted(Counter(detection_counts).items())
            ),
        },
        "tracks": {
            "total": sum(track_counts),
            "created": len(track_frames),
            "stats": calculate_track_length_stats(
                track_frames
            )
        },
        "track_lifetimes": track_lifetimes
    }


def analyze_track_lifetimes(frames):
    track_lifetimes = defaultdict(list)

    for frame in frames:
        frame_index = frame["frame_index"]

        for track in frame["tracks"]:
            track_id = track["track_id"]
            track_lifetimes[track_id].append(frame_index)

    lifetimes = {}

    for track_id, frame_indices in track_lifetimes.items():
        lifetimes[track_id] = {
            "start_frame": min(frame_indices),
            "end_frame": max(frame_indices),
            "length": len(frame_indices)
        }

    return lifetimes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tracking Evaluation Scripts"
    )

    parser.add_argument(
        "-f", "--file", type=str, help="File to analyze", required=True
    )
    args = parser.parse_args()

    print(f"Analyze file : {args.file}")

    results = analyze_tracking_file(args.file)

    print(results)

