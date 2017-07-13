import numpy as np
import scipy as sp
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Should enter copies of X_train and X_test
def feature_log1p(X_train, X_test, all_features=False, thresh=0.75):
    if not all_features:
        skewed_features = (X_train.apply(sp.stats.skew, axis=0) > thresh)
        X_train.loc[:, skewed_features] = np.log1p( X_train.loc[:, skewed_features] )
        X_test.loc[:, skewed_features] = np.log1p( X_test.loc[:, skewed_features] )
    else:
        X_train = np.log1p(X_train)
        X_test = np.log1p(X_test)
        
    return(X_train, X_test)


def feature_scale(X_train, X_test):
    numerical_columns = X_train.select_dtypes(include=['int64']).columns

    std_scaler = StandardScaler()

    X_train.loc[:, numerical_columns] = std_scaler.fit_transform(X_train.loc[:, numerical_columns])
    X_test.loc[:, numerical_columns] = std_scaler.transform(X_test.loc[:, numerical_columns])

    return(X_train, X_test)
