from abc import ABC, abstractmethod

import numpy as np
import torch
import torchreid

from PIL import Image
from torchvision import transforms


class PersonEmbedder(ABC):

    def __init__(self, model_name, embedding_dimension):
        pass

    @abstractmethod
    def compute(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract Embedding from cropped image.
        Args:
            image: RGB image as a Numpy array.
            bbox: Bounding box in pixel coordinates: (x_min, y_min, x_max, y_max).

        Returns:
            L2-normalized embedding vector.
        """
        raise NotImplementedError


class FakeEmbedder(PersonEmbedder):
    """Deterministic embedder used for tests."""

    def __init__(self, dimension: int = 128):
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension

    def compute(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:

        if image.ndim != 3:
            raise ValueError("image must be a 3D array")

        x_min, y_min, x_max, y_max = bbox

        if not (0 <= x_min < x_max <= image.shape[1] and 0 <= y_min < y_max <= image.shape[0]):
            raise ValueError("invalid bounding box")

        # deterministic vector for testing.
        embedding = np.ones(self.dimension, dtype=np.float32)

        # L2 normalization.
        embedding /= np.linalg.norm(embedding)
        return embedding


class OSNetEmbedder(PersonEmbedder):
    """OSNet-based person re-identification embedder."""

    def __init__(self, model_name: str = "osnet_x0_25", device: str = "cpu", image_size: tuple[int, int] = (256, 128)):
        self.device = torch.device(device)
        self.image_size = image_size

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1,
            pretrained=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]

        )

    def compute(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        if image.ndim != 3:
            raise ValueError("image must be a 3D array")

        x_min, y_min, x_max, y_max = bbox

        if not (0 <= x_min < x_max <= image.shape[1] and 0 <= y_min < y_max <= image.shape[0]):
            raise ValueError("invalid bounding box")

        # Numpy RGB to PIL image
        person_crop = image[y_min:y_max, x_min:x_max]

        if person_crop.size == 0:
            raise ValueError("empty peron crop")

        person_crop = Image.fromarray(person_crop)

        # Preprocessing
        tensor = self.transform(person_crop)

        # Add batch dimension
        tensor = tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            embedding = self.model(tensor)

        # Tensor to Numpy
        embedding = embedding.squeeze(0).cpu().numpy()

        # L2 normalization
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("model returned a zero embedding")

        embedding = embedding / norm

        return embedding.astype(np.float32)


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""

    if embedding_a.ndim != 1 or embedding_b.ndim != 1:
        raise ValueError("embeddings must be 1-dimensional")

    if embedding_a.shape != embedding_b.shape:
        raise ValueError("embeddings must have the same dimension")

    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError("embeddings must not be zero vectors")

    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


