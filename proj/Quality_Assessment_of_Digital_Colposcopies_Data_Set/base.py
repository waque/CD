import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
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
    
    

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res

def plot_results(clf, X_train, X_test, y_train, y_test, filename='result'):
    
    sns.set()
    sns.set_style("whitegrid")
    sns.despine()
    
    measures_dict = {}

    res = classifier_statistics(clf, X_train, X_test, y_train, y_test)
    accuracy = res['accuracy']
    sensibility = res['sensibility']
    specificity = res['specificity']
    measures_dict[0] = {'Measure': 'Accuracy', 'Value': accuracy}
    
    measures_dict[0] = {'Measure': 'Sensibility', 'Value': sensibility}
    
    measures_dict[0] = {'Measure': 'Specificity', 'Value': specificity}

    measures = pd.DataFrame.from_dict(measures_dict, "index")
    #measures.to_csv('/{}.csv'.format(filename))
    #plt.figure(figsize=figsize)
    ax = sns.barplot(x='Technique', y='Value', hue='Measure', data=measures)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.3f}'.format(float(p.get_height())), 
            fontsize=12, color='black', ha='center', va='bottom')
    
    plt.savefig('{}.pdf'.format(filename))
    plt.clf()

    return results

X_train, X_test, y_train, y_test, X_train_res, y_train_res = balance_dataset(data[0][0], 'consensus')

naive_cls = GaussianNB()

plot_results(naive_cls,X_train, X_test, y_train, y_test)