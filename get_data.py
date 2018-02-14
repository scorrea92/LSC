
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

def data_clean_X(path):
    test = genfromtxt(path, delimiter=',')
    test = test[1:,1:]
    # print(test.shape)

    impcons = test[:,:17]
    impsald = test[:,17:38]
    rest0 = test[:,38:62]
    numope = test[:,62:82]
    rest = test[:,82:83]
    rest1 = test[:,83:87]

    np.place(rest, np.isnan(rest), 10301)

    a = np.trunc(rest/1000)
    b = np.trunc(rest/100) - a*10
    c = np.trunc(rest/10)- (a*100 + b*10)
    d = rest - (a*1000 + b*100 +c*10)
    rest2 = np.column_stack((a, b, c, d))

    impcons = StandardScaler().fit_transform(impcons)
    impsald = StandardScaler().fit_transform(impsald)
    numope = StandardScaler().fit_transform(numope)
    np.place(rest0, rest0==2, 1)
    rest1 = StandardScaler().fit_transform(rest1)

    sub_total = np.column_stack((impcons, impsald, numope))
    
    pca = joblib.load('pca_impcons_impsald_numope.pkl')
    sub_total = pca.transform(sub_total) 

    sub_total = np.column_stack((sub_total, rest0, rest2, rest1))
    # print(sub_total)
    # print(sub_total.shape)
    return sub_total

def data_clean_Y(path):
    test = genfromtxt(path, delimiter=',')
    test = test[1:,1:]
    # print(test.shape)
    rest = test[:,87:]
    return rest

# a = data_clean_x('../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TRAIN.txt')
# a = data_clean_Y('../dataset_cajamar/Dataset_Salesforce_Predictive_Modelling_TRAIN.txt')
# print(a)
# print(a.shape)