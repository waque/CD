import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('..')
from utils_cd import (
        split_dataset_transformed,
        cross_val,
        print_dict,
        split_dataset,
        classifier_statistics
)
from sklearn.neighbors import KNeighborsClassifier

def pprint(res):
    print("\nthis classifier got the following results: \n")
    print("accuracy: " + str(res['accuracy']) + "\n")
    print("sensibility: " + str(res['sensibility']) + "\n")
    print("specificity: " + str(res['specificity']) + "\n")


aps_train = pd.read_csv('./aps_failure_training_set.csv',
                        skiprows=20,keep_default_na=False)


aps_test = pd.read_csv('./aps_failure_test_set.csv',
                        skiprows=20,keep_default_na=False)


aps_train.replace('na', np.nan, inplace=True)
aps_test.replace('na', np.nan, inplace=True)

#drop NaN values
#problema -> passamos de 60000 rows para apenas 591
aps_train = aps_train.dropna()

aps_test = aps_test.dropna()

X_train, y_train = split_dataset(aps_train, 'class')
X_test, y_test = split_dataset(aps_train, 'class')

y_train = y_train.map({'pos': 1, 'neg': 0})
y_test = y_test.map({'pos': 1, 'neg': 0})

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')


neigh = KNeighborsClassifier(n_neighbors=10)

res = classifier_statistics(neigh, X_train, X_test, y_train, y_test) 

pprint(res)

"""
k=2
accuracy: 0.936
sensibility: 0.5

k=3 
accuracy: 0.946
sensibility: 0.658

k=10
accuracy: 0.892
sensibility: 0.224

Analisar o trade-off sensibility vs accuracy
Se é preferível ter mais FP ou FN de a acordo com o contexto. 
Quanto maior for a Sensitivity, maior é o número de FP
Quanto menor for a Sensitivity, maior é o número de FN

specificity ~1 -> quase não existem FP
"""