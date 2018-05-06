import numpy as np
import get_data
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_data_own(path_train):

    train_x = get_data.data_clean_X(path_train)
    train_y = get_data.data_clean_Y(path_train)

    c = list(zip(train_x, train_y))
    random.shuffle(c)
    train_x, train_y = zip(*c)

    return np.array(train_x), np.array(train_y)

max_iter = 1000

# Get Data
print("Getting data")
path_train = 'data_train.txt'
path_test = 'data_test.txt'

X, Y = get_data_own(path_train)

print(X.shape)
print(Y.shape)

print("Split data for CV")
X_train, X_test , y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

lasso = MultiTaskLasso(max_iter = max_iter, normalize = True)

print("Init train with multitasklassocv")
lassocv = MultiTaskLassoCV(alphas=None, cv=10, max_iter=max_iter, verbose=True, normalize=True)
lassocv.fit(X_train, y_train)

print("Fit multitasklasso with alpha from cv lasso")
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)

print("get mean square error")
mae = mean_absolute_error(y_test, lasso.predict(X_test))
print("mae: {}".format(mae))
rmsle = mean_squared_log_error(y_test, lasso.predict(X_test))
print("rmsle: {}".format(rmsle))
mape = mean_absolute_percentage_error(y_test, lasso.predict(X_test))
print("mape: {}".format(mape))



