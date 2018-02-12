
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.externals import joblib

def getData_standard(path):
    imp_consum = genfromtxt(path, delimiter=',')
    x_std = StandardScaler().fit_transform(imp_consum)
    a = x_std
    return a

def get_total_pca(path, n_impcons, n_impsald, n_numope, pca_value, recall):
    impcons = getData_standard(path + n_impcons)
    impsald = getData_standard(path + n_impsald)
    numope = getData_standard(path + n_numope) 
    total = np.column_stack((impcons, impsald, numope))
    print(total.shape)
    
    if recall:
        pca = decomposition.PCA(n_components=pca_value, svd_solver='full')
        pca.fit(total)
        total_pca = pca.transform(total)
        joblib.dump(pca, 'pca_impcons_impsald_numope.pkl')
    else:
        pca = joblib.load('pca_impcons_impsald_numope.pkl')
        total_pca = pca.transform(total) 

    return total_pca

# def test_to_pca(test):

a = get_total_pca('../dataset_cajamar/', 'impcons.csv', 'impsald.csv', 'numope.csv', 0.95, True)

print(a.shape)
print(a)