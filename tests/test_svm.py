import numpy as np

from src.models.svm import LinearSVM


def test_svm_learns_separable_multiclass_data():
    rng = np.random.default_rng(1)
    x = np.vstack(
        [
            rng.normal((-2, -2), 0.2, (30, 2)),
            rng.normal((2, -2), 0.2, (30, 2)),
            rng.normal((0, 2), 0.2, (30, 2)),
        ]
    ).astype(np.float32)
    y = np.repeat(np.arange(3), 30)
    model = LinearSVM(learning_rate=0.05, epochs=30, batch_size=15, seed=1).fit(x, y)
    assert np.mean(model.predict(x) == y) > 0.98
    assert model.loss_history_[-1] < model.loss_history_[0]

