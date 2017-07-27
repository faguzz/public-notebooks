import numpy as np
import scipy as sp
import pandas as pd

import xgboost
from xgboost.sklearn import XGBRegressor

from sklearn.model_selection import GridSearchCV
from tuning_clf import model_fit, cv_report

datafilename = 'data_extracted/cs-training-prepared.csv'
targetvar = 'SeriousDlqin2yrs'


df = pd.read_csv(datafilename)
print('DataFrame shape: %d x %d' % (df.shape))

X = df.drop([targetvar], axis=1)
y = df[targetvar]
scoring = 'roc_auc' # ['accuracy', 'adjusted_rand_score', 'average_precision', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc']



xgb = XGBRegressor()

model_fit(xgb, X, y, classifier=False)



param_test1 = {'max_depth': range(3,10,2),
               'min_child_weight': range(1,6,2)}

gsearch1 = GridSearchCV(xgb, param_grid=param_test1, scoring=scoring,
                        n_jobs=1, iid=False, cv=5, verbose=2)

gsearch1.fit(X, y)

cv_report(gsearch1.cv_results_)



xgb = gsearch1.best_estimator_

param_test1b = {'max_depth': gsearch1.best_params_['max_depth'] + np.array([-1, 0, 1]),
                'min_child_weight': gsearch1.best_params_['min_child_weight'] + np.array([-1, 0, 1])}

gsearch1b = GridSearchCV(xgb, param_grid=param_test1b, scoring=scoring,
                        n_jobs=1, iid=False, cv=5, verbose=2)

gsearch1b.fit(X, y)

cv_report(gsearch1b.cv_results_)



xgb = gsearch1b.best_estimator_

param_test2 = {'gamma': [i/10.0 for i in range(0,5)]}

gsearch2 = GridSearchCV(xgb, param_grid=param_test2, scoring=scoring,
                        n_jobs=1, iid=False, cv=5, verbose=2)

gsearch2.fit(X, y)

cv_report(gsearch2.cv_results_)



xgb = gsearch2.best_estimator_

param_test3 = {'subsample': [i/10.0 for i in range(6,10)],
               'colsample_bytree': [i/10.0 for i in range(6,10)]}

gsearch3 = GridSearchCV(xgb, param_grid=param_test3, scoring=scoring,
                        n_jobs=1, iid=False, cv=5, verbose=2)

gsearch3.fit(X, y)

cv_report(gsearch3.cv_results_)



xgb = gsearch3.best_estimator_

param_test3b = {'subsample': np.linspace(gsearch3.best_params_['subsample'] - 0.1,
                                         gsearch3.best_params_['subsample'] + 0.1, 5),
                'colsample_bytree': np.linspace(gsearch3.best_params_['colsample_bytree'] - 0.1,
                                                gsearch3.best_params_['colsample_bytree'] + 0.1, 5)}

gsearch3b = GridSearchCV(xgb, param_grid=param_test3b, scoring=scoring,
                         n_jobs=1, iid=False, cv=5, verbose=2)
gsearch3b.fit(X, y)

cv_report(gsearch3b.cv_results_)



xgb = gsearch3b.best_estimator_

param_test4 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch4 = GridSearchCV(xgb, param_grid=param_test4, scoring=scoring,
                        n_jobs=1, iid=False, cv=5, verbose=2)

gsearch4.fit(X, y)

cv_report(gsearch4.cv_results_)



xgb.learning_rate = 0.1
xgb.n_estimators = 500

model_fit(xgb, X, y, classifier=False)



print(xgb)

