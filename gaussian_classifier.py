import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from sklearn.preprocessing import StandardScaler

import get_data

# Get Data
path_train = 'data_train.txt'
path_test = 'data_test.txt'

x_train, y_train, x_val, y_val, test = get_data.import_data(path_train, path_test)

print("x and y train shape")
print(x_train.shape)
print(y_train.shape)

print("x and y validation shape")
print(x_val.shape)
print(y_val.shape)

print(min(y_train))
print(max(y_train))

print(x_train.min())
print(x_train.max())

print("y before")
print(y_train)

data = np.column_stack((x_train,y_train))

scaler = StandardScaler()
scaler.fit(data)
new_y = scaler.transform(data)

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=4, covariance_type='full').fit(new_y)

# Fit a Dirichlet process Gaussian mixture using five components
#dpgmm = mixture.BayesianGaussianMixture(n_components=3, covariance_type='full',max_iter=1000).fit(new_y)

print("Covariances")
print(gmm.covariances_)
print("Means Shape")
print(gmm.means_.shape)
print("Means ")
print(gmm.means_)
print("Means Transformed")
print(scaler.inverse_transform(gmm.means_))
np.savetxt("means.txt", gmm.means_, delimiter=",")
