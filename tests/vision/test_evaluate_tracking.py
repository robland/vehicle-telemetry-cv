import pytest

import json
from collections import Counter


from scripts.evaluate_tracking import calculate_track_length_stats, analyze_tracking_file, analyze_track_lifetimes


def test_calculate_track_length_stats_with_empty_track_frames():
    track_frames = {}
    stats = calculate_track_length_stats(track_frames)

    assert stats["count"] == 0


def test_calculate_track_length_stats_with_valid_track_frames():
    track_frames = Counter({1: 5, 2: 25, 3: 14, 4: 5})
    stats = calculate_track_length_stats(track_frames)

    assert stats["count"] == 4
    assert stats["min"] == 5
    assert stats["max"] == 25
    assert stats["mean"] == pytest.approx(12.25)
    assert stats["median"] == pytest.approx(9.5)
    assert stats["p25"] == pytest.approx(5)
    assert stats["p75"] == pytest.approx(14)
    assert stats["p90"] == pytest.approx(21.7)
    assert stats["track_ge_10"] == 2
    assert stats["track_ge_30"] == 0
    assert stats["track_ge_100"] == 0
    assert stats["track_ge_500"] == 0


def write_fake_jsonl(tracking_file_name):
    tracking_file_content = [
        {
            "frame_index": 0,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.90,
                    "bbox": {
                        "x_min": 0.10,
                        "y_min": 0.10,
                        "x_max": 0.30,
                        "y_max": 0.50,
                    },
                },
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.85,
                    "bbox": {
                        "x_min": 0.50,
                        "y_min": 0.10,
                        "x_max": 0.70,
                        "y_max": 0.50,
                    },
                },
            ],
            "tracks": [
                {
                    "track_id": 1,
                    "bbox": {
                        "x_min": 0.10,
                        "y_min": 0.10,
                        "x_max": 0.30,
                        "y_max": 0.50,
                    },
                    "confidence": 0.90,
                },
                {
                    "track_id": 2,
                    "bbox": {
                        "x_min": 0.50,
                        "y_min": 0.10,
                        "x_max": 0.70,
                        "y_max": 0.50,
                    },
                    "confidence": 0.85,
                },
            ],
        },
        {
            "frame_index": 1,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.92,
                    "bbox": {
                        "x_min": 0.11,
                        "y_min": 0.10,
                        "x_max": 0.31,
                        "y_max": 0.50,
                    },
                },
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.87,
                    "bbox": {
                        "x_min": 0.49,
                        "y_min": 0.10,
                        "x_max": 0.69,
                        "y_max": 0.50,
                    },
                },
            ],
            "tracks": [
                {
                    "track_id": 1,
                    "bbox": {
                        "x_min": 0.11,
                        "y_min": 0.10,
                        "x_max": 0.31,
                        "y_max": 0.50,
                    },
                    "confidence": 0.92,
                },
                {
                    "track_id": 2,
                    "bbox": {
                        "x_min": 0.49,
                        "y_min": 0.10,
                        "x_max": 0.69,
                        "y_max": 0.50,
                    },
                    "confidence": 0.87,
                },
            ],
        },
        {
            "frame_index": 2,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.91,
                    "bbox": {
                        "x_min": 0.12,
                        "y_min": 0.10,
                        "x_max": 0.32,
                        "y_max": 0.50,
                    },
                },
            ],
            "tracks": [
                {
                    "track_id": 1,
                    "bbox": {
                        "x_min": 0.12,
                        "y_min": 0.10,
                        "x_max": 0.32,
                        "y_max": 0.50,
                    },
                    "confidence": 0.91,
                },
            ],
        },
    ]

    with open(tracking_file_name, "w", encoding="utf-8") as file:
        for frame in tracking_file_content:
            file.write(json.dumps(frame) + "\n")


def test_analyze_tracking_file(tmp_path):

    tracking_file_name = tmp_path / "tracking_file.jsonl"
    write_fake_jsonl(tracking_file_name)
    result = analyze_tracking_file(tracking_file_name)

    assert result["frames"]["total"] == 3
    assert result["tracks"]["created"] == 2

    stats = result["tracks"]["stats"]

    assert stats["count"] == 2
    assert stats["min"] == 2
    assert stats["max"] == 3
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)


def test_analyze_track_lifetimes():
    frames = [
        {
            "frame_index": 10,
            "tracks": [
                {"track_id": 1},
                {"track_id": 2},
            ],
        },
        {
            "frame_index": 11,
            "tracks": [
                {"track_id": 1},
                {"track_id": 2},
            ],
        },
        {
            "frame_index": 12,
            "tracks": [
                {"track_id": 1},
            ],
        },
        {
            "frame_index": 13,
            "tracks": [
                {"track_id": 1},
            ],
        },
    ]
    lifetimes = analyze_track_lifetimes(frames)
    assert lifetimes[1] == {
        "start_frame": 10,
        "end_frame": 13,
        "length": 4
    }

    assert lifetimes[2] == {
        "start_frame": 10,
        "end_frame": 11,
        "length": 2
    }