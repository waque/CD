import pandas as pd
import numpy as np
import collections
from data_exploration import get_data
from imblearn.over_sampling import SMOTE
import sys
sys.path.append('..')
from utils_cd import (
        split_train_test,
        classifier_statistics,
        print_dict,
        split_dataset
)
from sklearn.neighbors import KNeighborsClassifier

X, X_test, y, y_test = get_data(False)
#X_train, X_test, y_train, y_test = split_train_test(X, y)

"""
Balancing data only with training instances
so we can check if it generalizes well on the
unbalanced set with X_test and y_test
"""

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X, y)

print('Balanced data: {}'.format(collections.Counter(y_train_res)))
"""
clf = RandomForestClassifier(n_estimators=25, random_state=42)
results = classifier_statistics(clf, X_train_res, X_test, y_train_res, y_test)

print_dict(results, excluded_keys=['predicted'])

cost1 = 10
cost2 = 500
conf_matrix = results['confusion_matrix']
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]

total_cost = cost1*fp + cost2*fn

print('Total cost aachived: {}'.format(total_cost))
"""

"""
probably something is wrong here, the results seem to be too good
"""


clf = KNeighborsClassifier(n_neighbors=10)
results = classifier_statistics(clf, X_train_res, X_test, y_train_res, y_test)

print_dict(results, excluded_keys=['predicted'])

cost1 = 10
cost2 = 500
conf_matrix = results['confusion_matrix']
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]

total_cost = cost1*fp + cost2*fn

print('Total cost aachived: {}'.format(total_cost))