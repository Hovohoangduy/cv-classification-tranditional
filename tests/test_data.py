import numpy as np

from src.data.cifar10 import stratified_split


def test_stratified_split_is_disjoint_and_balanced():
    labels = np.repeat(np.arange(3), 20)
    train, validation = stratified_split(labels, 0.2, seed=7)
    assert len(set(train) & set(validation)) == 0
    assert sorted(np.bincount(labels[validation]).tolist()) == [4, 4, 4]
    assert len(train) + len(validation) == len(labels)

