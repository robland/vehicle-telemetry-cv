from dataclasses import dataclass


@dataclass
class PersonDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: dict[str, float]

    def __post_init__(self):
        if not isinstance(self.class_id, int):
            raise TypeError("Expected type int for class_id.")

        if not isinstance(self.class_name, str):
            raise TypeError("Expected type str for class_name.")

        if not self.class_name == "person":
            raise ValueError(f"Expected value person but got {self.class_name}.")

        if not isinstance(self.confidence, int | float):
            raise TypeError(f"Expected number for confidence.")

        if not (0 <= self.confidence <= 1.0):
            raise ValueError(f"Expected confidence between [0, 1] but got {self.confidence}.")

        if self.class_id < 0:
            raise ValueError(f"Expected class_id >= 0 but got {self.class_id}.")

        if self.bbox.get("x_min") is None:
            raise ValueError(f"Expected x_min value.")

        if self.bbox.get("x_max") is None:
            raise ValueError(f"Expected x_max value.")

        if self.bbox.get("y_min") is None:
            raise ValueError(f"Expected y_min value.")

        if self.bbox.get("y_max") is None:
            raise ValueError(f"Expected y_max value.")

        if not (0 <= self.bbox["x_min"] < self.bbox["x_max"] <= 1):
            raise ValueError("Expected value with constraint 0 <= x_min < x_max <= 1.")

        if not (0 <= self.bbox["y_min"] < self.bbox["y_max"] <= 1):
            raise ValueError("Expected value with constraint 0 <= y_min < y_max <= 1.")


