import json

import cv2
import numpy as np

from monitoring_taxi.vision.embedding import cosine_similarity, OSNetEmbedder


def evaluate_embedding(source: str, track_file_name: str) -> dict:
    """We want to evaluate OSNet with truth ground images.

    Args:
        source: pathname video as a string
        track_file_name: pathname jsonl track file as string
    Returns:
        None
    """
    start_frame, end_frame = (2031, 2538)

    cap = cv2.VideoCapture(source)
    embedder = OSNetEmbedder()

    same_person = []
    different_person = []

    embeddings = {}

    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}.")

    try:
        with open(track_file_name, "r", encoding="utf-8") as tracking_file:

            for line in tracking_file:

                data = json.loads(line)

                ret, frame = cap.read()
                if not ret:
                    break

                if not start_frame <= data["frame_index"] < end_frame:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_height, image_width, _ = frame_rgb.shape
                tracks = data["tracks"]

                new_embeddings = {}

                for track in tracks:

                    track_id = track["track_id"]
                    bbox = track["bbox"]

                    embedding = embedder.compute(
                        frame_rgb,
                        (
                            int(bbox["x_min"] * image_width),
                            int(bbox["y_min"] * image_height),
                            int(bbox["x_max"] * image_width),
                            int(bbox["y_max"] * image_height)
                        )
                    )

                    new_embeddings[track_id] = embedding

                if embeddings:

                    for track_id_a, embedding_a in embeddings.items():

                        for track_id_b, embedding_b in new_embeddings.items():
                            similarity = cosine_similarity(
                                embedding_a,
                                embedding_b
                            )

                            if track_id_a == track_id_b:
                                same_person.append(similarity)
                            else:
                                different_person.append(similarity)

                embeddings = new_embeddings

    finally:
        cap.release()

    return {
        "same_person": same_person,
        "different_person": different_person
    }


def evaluate_threshold(
        same_person: list[float],
        different_person: list[float],
        threshold: float,
) -> dict:
    """Evaluate an embedding similarity threshold."""

    same_person = np.asarray(same_person)
    different_person = np.asarray(different_person)

    same_accepted = np.sum(same_person >= threshold)
    same_rejected = np.sum(same_person < threshold)

    different_accepted = np.sum(different_person >= threshold)
    different_rejected = np.sum(different_person < threshold)

    same_count = len(same_person)
    different_count = len(different_person)

    fnmr = same_rejected / same_count if same_count else 0.0
    fmr = different_accepted / different_count if different_count else 0.0

    tpr = same_accepted / same_count if same_count else 0.0
    tnr = different_rejected / different_count if different_count else 0.0

    return {
        "threshold": threshold,
        "fnmr": fnmr,
        "fmr": fmr,
        "tpr": tpr,
        "tnr": tnr,
        "same_accepted": int(same_accepted),
        "same_rejected": int(same_rejected),
        "different_accepted": int(different_accepted),
        "different_rejected": int(different_rejected),
    }


def evaluate_thresholds(
        same_person: list[float],
        different_person: list[float],
) -> list[dict]:

    thresholds = np.arange(0.50, 0.951, 0.01)

    return [
        evaluate_threshold(
            same_person,
            different_person,
            float(threshold),
        )
        for threshold in thresholds
    ]


if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser("Evaluate embedding script")
    parser.add_argument(
        "-s", "--source", type=str, help="Video source", required=True
    )

    parser.add_argument(
        "-f", "--file", type=str, help="File to Analyze", required=True
    )
    args = parser.parse_args()

    print(f"Analyze file : {args.file} from Video {args.source}")

    results = evaluate_embedding(args.source, args.file)

    print(pd.DataFrame(results['same_person']).describe())
    print(pd.DataFrame(results['different_person']).describe())

    results = evaluate_thresholds(results['same_person'], results['different_person'])
    pd_results = pd.DataFrame(results)

    print("Statistique Descriptive")
    print(pd_results)












