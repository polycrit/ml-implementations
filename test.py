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

ds = linear_separable()
print(ds.X.shape, ds.y.shape)
predict_fn = train_perceptron_2d(ds.X, ds.y, 1000)
plot_dataset(ds)
plot_decision_boundary(ds, predict_fn)
plt.show()
