from generate_dataset_2d import (
    linear_separable,
    blobs,
    moons,
    circles,
    plot_dataset,
    plot_decision_boundary,
)
import matplotlib.pyplot as plt
from perceptron import train_perceptron_2d
from regression import train_lr_2d

ds = linear_separable()
print(ds.X.shape, ds.y.shape)
predict_fn = train_lr_2d(ds.X, ds.y, 0.001, 0.001, 0.00001, 100_000_000)
plot_decision_boundary(ds, predict_fn)
plt.show()
