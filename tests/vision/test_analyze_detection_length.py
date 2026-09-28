import pytest

from monitoring_taxi.analyze_utils import calculate_length_loss


def test_calculate_loss_detection():
    lengths = [2, 2, 3, 1, 1, 3, 1]
    losses = calculate_length_loss(lengths)

    assert len(losses) == 3
    assert losses == [2, 2, 1]

