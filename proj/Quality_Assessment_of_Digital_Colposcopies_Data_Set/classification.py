import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
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
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

green_data = pd.read_csv('./green.csv')
hinselmann_data = pd.read_csv('./hinselmann.csv')
schiller_data = pd.read_csv('./schiller.csv')

data = [[green_data,'green_data'], [hinselmann_data,'hinselmann_data'], [schiller_data,'schiller_data']]


def balance_dataset(dataset, classe):
	y = dataset[classe]
	X = dataset.drop(columns=[classe])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	sm = SMOTE(random_state=2)

	X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
	
	#print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
	#print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

	#print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
	#print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

	#print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
	#print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

	return X_train, X_test, y_train, y_test, X_train_res, y_train_res

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
		X_train, X_test, y_train, y_test, X_train_res, y_train_res = balance_dataset(d[0], 'consensus')
		res = classifier_statistics(gnb, X_train_res, X_test, y_train_res.ravel(), y_test.ravel())
		export_file(res, d[1], 'Naive bayes balenced', "")

