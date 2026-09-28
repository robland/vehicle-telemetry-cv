import numpy as np


def calculate_length_loss(lengths: list[int], expected_detections: int = 3):
    losses = []
    loss = 0

    for length in lengths:
        if length < expected_detections:
            loss += 1
        elif loss > 0:
            losses.append(loss)
            loss = 0

    if loss > 0:
        losses.append(loss)

    return losses


def calculate_stats(losses: list[int]):

    return dict(
        min=min(losses),
        max=max(losses),
        sum=sum(losses),
        mean=sum(losses)/len(losses),
        median=np.median(losses)
    )


def analyze_length_inter_detection(length_detections: list[int], start: int = 67, end: int | None = 2911):
    if not length_detections:
        raise ValueError("length_detections cannot be empty")

    results = []
    current_value = length_detections[start]
    length = 1

    for current in length_detections[start+1:end+1]:
        if current == current_value:
            length += 1
        else:
            results.append(
                {
                    "detections": current_value,
                    "length": length,
                }
            )

            current_value = current
            length = 1

    # Dernière séquence
    results.append(
        {
            "detections": current_value,
            "length": length,
        }
    )

    lengths = [result["length"] for result in results]

    return {
        "sequences": results,
        "stats": {
            "max": max(lengths),
            "min": min(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": np.median(lengths),
            "count": len(lengths),
        }
    }
