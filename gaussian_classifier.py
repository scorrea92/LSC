import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn.preprocessing import StandardScaler

import get_data

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
# Get Data
n = 3 # Number of Gausssians
path_train = 'data_train.txt'
path_test = 'data_test.txt'

print("Importing data ...")
x_train, y_train, x_val, y_val, test = get_data.import_nopca(path_train, path_test)

print("x and y train shape")
print(x_train.shape)
print(y_train.shape)

print("x and y validation shape")
print(x_val.shape)
print(y_val.shape)

print("min and max of purchasing power")
print(min(y_train))
print(max(y_train))

# Stack x_train and y_train
# data = np.column_stack((x_train,y_train)) # Todos
data = np.column_stack((x_train[:,59:],y_train)) # Imp Consumo + PA

# Scaling data
scaler = StandardScaler()
scaler.fit(data)
data_tr = scaler.transform(data)
# Fit a Gaussian mixture with EM using five components
print("Fitting GMM ...")
gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(data_tr)
# Fit a Dirichlet process Gaussian mixture using five components
#dpgmm = mixture.BayesianGaussianMixture(n_components=n, covariance_type='full',max_iter=1000).fit(scaler.transform(data))

print("Covariances")
print(gmm.covariances_)
print("Means Shape")
print(gmm.means_.shape)
print("Means ")
print(gmm.means_)
print("Means Transformed")
means_tr = scaler.inverse_transform(gmm.means_)
print(means_tr[:, -1])
# np.savetxt("means.txt", gmm.means_, delimiter=",")

#print("Predicting ...")
#n_predicts = 1000
#labels = gmm.predict(data_tr[0:n_predicts,:])
#plt.scatter(data_tr[0:n_predicts, 0], data_tr[0:n_predicts, 1], c=labels, s=40, cmap='viridis')
#plt.show()