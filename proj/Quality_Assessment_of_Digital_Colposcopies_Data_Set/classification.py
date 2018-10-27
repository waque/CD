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
	export_file,
	balance_dataset
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)


green_data = pd.read_csv('./green.csv')
hinselmann_data = pd.read_csv('./hinselmann.csv')
schiller_data = pd.read_csv('./schiller.csv')

data = [[green_data,'green_data'], [hinselmann_data,'hinselmann_data'], [schiller_data,'schiller_data']]


def knn ():
    n_neighbors_values = [2,3,10]
    for d in data:
		for n in n_neighbors_values:
			X, y = split_dataset_transformed(d[0], 'consensus')
			X_train, X_test, y_train, y_test = split_train_test(X, y)
			neigh = KNeighborsClassifier(n_neighbors=n)
			res = classifier_statistics(neigh, X_train, X_test, y_train, y_test) 
			export_file(res, d[1], 'KNN', "k=" + str(n))

def naive_bayes():
	gnb = GaussianNB()
	for d in data:
		X, y = split_dataset_transformed(d[0], 'consensus')
		X_train, X_test, y_train, y_test = split_train_test(X, y)
		res = classifier_statistics(gnb, X_train, X_test, y_train, y_test) 
		export_file(res, d[1], 'Naive bayes', "")


def naive_bayes_balenced():
	gnb = GaussianNB()
	for d in data:
		X_train, X_test, y_train, y_test, X_train_res, y_train_res = balance_dataset(d, 'consensus')
		res = classifier_statistics(gnb, X_train_res, X_test_res, y_train, y_test) 
		export_file(res, d[1], 'Naive bayes balenced', "")


naive_bayes_balenced()
