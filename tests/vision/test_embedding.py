import numpy as np
import pytest

from src.monitoring_taxi.vision.embedding import FakeEmbedder, cosine_similarity, OSNetEmbedder


def test_fake_embedder_dimension():
    image = np.zeros((100, 100, 3), np.uint8)
    bbox = (10, 10, 50, 80)

    embedder = FakeEmbedder(dimension=128)

    embedding = embedder.compute(image, bbox)

    assert embedding.shape == (128, )
    assert embedding.dtype == np.float32


def test_fake_embedder_is_l2_normalized():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10, 10, 50, 80)

    embedder = FakeEmbedder(dimension=128)
    embedding = embedder.compute(image, bbox)

    assert np.isclose(np.linalg.norm(embedding), 1.0)


def test_fake_embedder_is_deterministic():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10, 10, 50, 80)

    embedder = FakeEmbedder(dimension=128)

    embedding_1 = embedder.compute(image, bbox)
    embedding_2 = embedder.compute(image, bbox)

    assert np.array_equal(embedding_1, embedding_2)


def test_fake_embedder_rejects_invalid_image():
    image = np.zeros((100, 100), dtype=np.uint8)
    bbox = (10, 10, 50, 80)

    embedder = FakeEmbedder()

    with pytest.raises(ValueError):
        embedder.compute(image, bbox)


@pytest.mark.parametrize(
    "bbox",
    [
        (-1, 10, 50, 80),
        (10, -1, 50, 80),
        (10, 10, 101, 80),
        (10, 10, 50, 101),
        (50, 10, 10, 80),
        (10, 80, 50, 10)
    ]

)
def test_fake_embedder_rejects_invalid_bbox(bbox):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    embedder = FakeEmbedder()

    with pytest.raises(ValueError):
        embedder.compute(image, bbox)


def test_cosine_similarity_identical_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])

    similarity = cosine_similarity(a, b)

    assert np.isclose(similarity, -1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    similarity = cosine_similarity(a, b)

    assert np.isclose(similarity, 0.0)


def test_cosine_similarity_same_direction_different_magnitude():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0, 30.0])

    similarity = cosine_similarity(a, b)

    assert np.isclose(similarity, 1.0)


def test_cosine_similarity_rejects_different_dimensions():
    a = np.array([1., 0., 0.0])
    b = np.array([1.0, 0.0])

    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_similarity_rejects_non_1d_embeddings():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_cosine_similarity_rejects_zero_vector():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_osnet_embedder_output():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    bbox = (50, 20, 150, 220)

    embedder = OSNetEmbedder()

    embedding = embedder.compute(image, bbox)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1.0)