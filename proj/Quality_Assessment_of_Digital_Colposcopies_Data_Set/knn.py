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
        classifier_statistics,
        pprint,
        split_train_test,
        export_file
)
from sklearn.neighbors import KNeighborsClassifier

green_data = pd.read_csv('./green.csv')
hinselmann_data = pd.read_csv('./hinselmann.csv')
schiller_data = pd.read_csv('./schiller.csv')

data = [[green_data,'green_data'], [hinselmann_data,'hinselmann_data'], [schiller_data,'schiller_data']]
n_neighbors_values = [2,3,10]


for d in data:
    for n in n_neighbors_values:
        X, y = split_dataset_transformed(d[0], 'consensus')
        X_train, X_test, y_train, y_test = split_train_test(X, y)
        neigh = KNeighborsClassifier(n_neighbors=n)
        res = classifier_statistics(neigh, X_train, X_test, y_train, y_test) 
        export_file(res, d[1], 'KNN', "k=" + str(n))


