import json
from collections import Counter
from datetime import datetime

from monitoring_taxi.analyze_utils import analyze_length_inter_detection
from monitoring_taxi.analyze_utils import calculate_length_loss
from monitoring_taxi.analyze_utils import calculate_stats

str_current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
TRACKING_SOURCE = "data/tracking/tracking_output_v0_2026-09-19 09-21-07.jsonl"
ANALYSIS_OUTPUT = f"data/analysis/tracking_output_v0_{str_current_datetime}.jsonl"


def analyse_tracking_output_file():
    frames = []

    with open(TRACKING_SOURCE, "r", encoding="utf-8") as tracking_file:
        for line in tracking_file:
            data = json.loads(line)

            frames.append(
                {
                    "frame": data["frame_index"],
                    "detections": len(data["detections"]),
                    "tracks": len(data["tracks"])
                }
            )

    detection_distribution = Counter(
        frame["detections"] for frame in frames
    )
    detection_variations = []
    results = analyze_length_inter_detection([frame["detections"] for frame in frames])

    for previous, current in zip(frames, frames[1:]):

        if previous["detections"] != current["detections"]:
                detection_variations.append(
                    {
                        "frame_from": previous["frame"],
                        "frame_to": current["frame"],
                        "detections_from": previous["detections"],
                        "detections_to": current["detections"],
                        "variation": (
                            current["detections"] - previous["detections"]
                        )
                    }
                )

    analysis = {
        "summary": {
            "total_frames": len(frames),
            "detection_distribution": dict(
                sorted(detection_distribution.items())
            ),
        },
        "detection_variations": detection_variations
    }

    with open(ANALYSIS_OUTPUT, "w", encoding="utf-8") as analysis_file:
        json.dump(
            analysis,
            analysis_file,
            indent=2
        )
        print(f"Analysis written to: {ANALYSIS_OUTPUT}")
        print("\n=== SUMMARY ===")
        print(f"Total frames: {len(frames)}")
        print("\nDetection distribution:")

        for detections, count in sorted(detection_distribution.items()):
            print(f"{detections} detection(s): " f"{count} frame(s)")
        print("\nDetection variations:")

        for variation in detection_variations:
            print(f"Frame "
                  f"{variation['frame_from']} → "
                  f"{variation['frame_to']}: " 
                  f"{variation['detections_from']} → "
                  f"{variation['detections_to']} "
            )

        print(" ############################# Length detections stats #############################")
        for sequence in results["sequences"]:
            print(f"{sequence['detections']} détections pendant {sequence['length']} frames")
        print(results["stats"])

        print(" ############################# Losses detections #############################")
        losses = calculate_length_loss([result["length"] for result in results["sequences"]])
        print(calculate_stats(losses))


if __name__ == "__main__":
    analyse_tracking_output_file()

