import numpy as np

from src.models.decision_tree import DecisionTreeClassifier
from src.models.random_forest import RandomForestClassifier


def test_decision_tree_learns_threshold():
    x = np.arange(20, dtype=np.float32)[:, None]
    y = (x[:, 0] >= 10).astype(np.int64)
    tree = DecisionTreeClassifier(max_depth=2, max_features=1, seed=1).fit(x, y)
    assert np.mean(tree.predict(x) == y) == 1.0


def test_random_forest_predicts_valid_classes():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    forest = RandomForestClassifier(num_trees=5, max_depth=5, seed=2).fit(x, y)
    prediction = forest.predict(x)
    assert set(np.unique(prediction)) <= {0, 1}
    assert np.mean(prediction == y) > 0.8

