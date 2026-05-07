import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

data_path = "./array_data/ccpp/ccpp.npz"
data_arr = np.load(data_path, allow_pickle = True)
x = data_arr["x"]
y = data_arr["y"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2000,random_state=21)

gbdt_estimator = HistGradientBoostingRegressor(learning_rate=0.05, random_state=5, 
                                          max_iter=80, max_depth=None, l2_regularization=1e-3, 
                                          early_stopping=True, n_iter_no_change=15, min_samples_leaf=50, validation_fraction=0.1,)
mlp_estimator = MLPRegressor(hidden_layer_sizes=[32, 128, 32], random_state=9)

gbdt_estimator.fit(x_train, y_train.ravel())
mlp_estimator.fit(x_train, y_train.ravel())

y_hat = gbdt_estimator.predict(x_test)
y_hat_nn = mlp_estimator.predict(x_test)

residuals = y_hat - y_test.ravel()
residuals_nn =  y_hat_nn - y_test.ravel()
residuals_total = y.ravel() - gbdt_estimator.predict(x)
residuals_total_nn = y.ravel() - mlp_estimator.predict(x)

import matplotlib.pyplot as plt

plt.hist(residuals_total, bins=50)
plt.savefig("./definitive_plots/residuals_tree_full_dataset.pdf")


plt.hist(residuals_total_nn, bins=50)
plt.savefig("./definitive_plots/residuals_nn_full_dataset.pdf")