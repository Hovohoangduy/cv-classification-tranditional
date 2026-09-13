import numpy as np

from src.metrics import classification_metrics, confusion_matrix


def test_confusion_matrix_orientation():
    matrix = confusion_matrix(np.array([0, 0, 1]), np.array([0, 1, 1]), 2)
    assert matrix.tolist() == [[1, 1], [0, 1]]


def test_perfect_metrics():
    result = classification_metrics(np.arange(4), np.arange(4), 4)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0

