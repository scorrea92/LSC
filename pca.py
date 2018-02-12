
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

imp_consum = genfromtxt('../dataset_cajamar/imp_consum.csv', delimiter=',')

x_std = StandardScaler().fit_transform(imp_consum)
# cov = np.cov(x_std.T)
# ev , eig = np.linalg.eig(cov)
# a = eig.dot(x_std.T)

pca = decomposition.PCA(n_components=0.95, svd_solver='full')
pca.fit(x_std)
a = pca.transform(x_std)

print(a.shape)
print(a)