from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, make_blobs, make_moons, make_circles


@dataclass
class Dataset2D:
    X: np.ndarray
    y: np.ndarray
    name: str = "dataset"
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "Dataset2D":
        return Dataset2D(self.X.copy(), self.y.copy(), self.name, self.meta)


def _to_pm1(y01: np.ndarray) -> np.ndarray:
    y01 = np.asarray(y01, dtype=int).ravel()
    return 2 * y01 - 1


def _validate_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X must be shape (n, 2). Got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"y must be shape (n,). Got {y.shape} vs X {X.shape}")

    uniq = set(np.unique(y).tolist())
    if not uniq.issubset({-1, 1}):
        raise ValueError(f"y must contain only -1 and 1. Got {sorted(uniq)}")

    return X, y


# ---------------------------
# scikit-learn generators
# ---------------------------


def linear_separable(
    n: int = 2000, class_sep: float = 3.0, seed: Optional[int] = 42
) -> Dataset2D:
    X, y01 = make_classification(
        n_samples=n,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=class_sep,
        flip_y=0.0,
        random_state=seed,
    )
    y = _to_pm1(y01)
    return Dataset2D(X, y, name="linear_separable", meta={"class_sep": class_sep})


def blobs(
    n: int = 2000,
    centers: Tuple[Tuple[float, float], Tuple[float, float]] = (
        (-2.0, -2.0),
        (2.0, 2.0),
    ),
    std: float = 0.8,
    seed: Optional[int] = 42,
) -> Dataset2D:
    X, y01 = make_blobs(
        n_samples=n,
        centers=np.array(centers, dtype=float),
        cluster_std=std,
        random_state=seed,
    )
    y = _to_pm1(y01)
    return Dataset2D(X, y, name="blobs", meta={"centers": centers, "std": std})


def moons(n: int = 2000, noise: float = 0.12, seed: Optional[int] = 42) -> Dataset2D:
    X, y01 = make_moons(n_samples=n, noise=noise, random_state=seed)
    y = _to_pm1(y01)
    return Dataset2D(X, y, name="moons", meta={"noise": noise})


def circles(
    n: int = 2000, factor: float = 0.5, noise: float = 0.08, seed: Optional[int] = 42
) -> Dataset2D:
    X, y01 = make_circles(n_samples=n, factor=factor, noise=noise, random_state=seed)
    y = _to_pm1(y01)
    return Dataset2D(X, y, name="circles", meta={"factor": factor, "noise": noise})


# ---------------------------
# Visualization
# ---------------------------


def plot_dataset(
    ds: Dataset2D,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    legend: bool = True,
    s: float = 12.0,
) -> plt.Axes:
    X, y = _validate_xy(ds.X, ds.y)
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(X[y == -1, 0], X[y == -1, 1], s=s, label="-1")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, label="+1")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or ds.name)
    if legend:
        ax.legend()
    return ax


def plot_decision_boundary(
    ds: Dataset2D,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    ax: Optional[plt.Axes] = None,
    grid_step: float = 0.03,
    padding: float = 0.6,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    predict_fn: (m, 2) -> (m,) labels in {-1, +1} or {0, 1}
    """
    X, y = _validate_xy(ds.X, ds.y)
    if ax is None:
        _, ax = plt.subplots()

    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, grid_step), np.arange(y_min, y_max, grid_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = np.asarray(predict_fn(grid)).astype(int).ravel()
    uniq = set(np.unique(zz).tolist())
    if uniq.issubset({0, 1}):
        zz = _to_pm1(zz)
    zz = zz.reshape(xx.shape)

    ax.contourf(xx, yy, zz, alpha=0.25)
    plot_dataset(ds, ax=ax, legend=True)
    ax.set_title(title or f"{ds.name} + decision boundary")
    return ax
