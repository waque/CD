import pandas as pd
import numpy as np
import collections
from data_exploration import get_data
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('..')
from utils_cd import (
        split_train_test,
        classifier_statistics,
        print_dict,
        split_dataset
)

X, y = get_data()
X_train, X_test, y_train, y_test = split_train_test(X, y)

"""
Balancing data only with training instances
so we can check if it generalizes well on the
unbalanced set with X_test and y_test
"""

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

print('Balanced data: {}'.format(collections.Counter(y_train_res)))

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
probably something is wrong here, the results seem to be too good
"""

aps = pd.read_csv('./aps_failure_test_set.csv',
                            skiprows=20,keep_default_na=False)
aps.replace('na', np.nan, inplace=True)
columns_to_remove = ['br_000', 'bq_000', 'bp_000', 'bo_000', 'ab_000', 'cr_000', 'bn_000', 'bm_000', 'cd_000']
aps = aps.drop(columns=columns_to_remove)
aps = aps.dropna()
X, y = split_dataset(aps, 'class')
X = X.astype('float64')
y = y.map({'pos': 1, 'neg': 0})

clf = RandomForestClassifier(n_estimators=25, random_state=42)
results = classifier_statistics(clf, X_train_res, X, y_train_res, y)

print_dict(results, excluded_keys=['predicted'])

cost1 = 10
cost2 = 500
conf_matrix = results['confusion_matrix']
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]

total_cost = cost1*fp + cost2*fn

print('Total cost aachived: {}'.format(total_cost))