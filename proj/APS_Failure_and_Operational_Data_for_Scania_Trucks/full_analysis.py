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
        standard_deviation
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.impute import SimpleImputer

CLASS = 'class'

orig_train = pd.read_csv('./aps_failure_training_set.csv',
                        skiprows=20,keep_default_na=False, na_values='na')
orig_test = pd.read_csv('./aps_failure_test_set.csv',
                        skiprows=20,keep_default_na=False, na_values='na')

"""

Let's first analyze the baseline we are working with,
we are going to evaluate Naive Bayes, Knn and Random Forest
with no preprocessing and checck if the results between them 
are similar or not.

"""

imp = SimpleImputer(strategy="constant", fill_value=0)

X_train, y_train = split_dataset(orig_train, CLASS)
X_test, y_test = split_dataset(orig_test, CLASS)
X_train = imp.fit_transform(X_train)
X_test = imp.fit_transform(X_test)

y_train = y_train.map({'pos': 1, 'neg': 0})
y_test = y_test.map({'pos': 1, 'neg': 0})

neighbors = [3, 5, 10]
estimators = [25, 50, 100]

naive_bayes = [(GaussianNB(), 'Gaussian'), (MultinomialNB(), 'Multinomial'), (BernoulliNB(), 'Bernoulli')]
knns = [(KNeighborsClassifier(n_neighbors=x), '{}'.format(x)) for x in neighbors]
random_forests = [(RandomForestClassifier(n_estimators=x), '{}'.format(x)) for x in estimators]

classifiers = {'Naive Bayes': naive_bayes, 'K Nearest Neighbors': knns, 'Random Forest': random_forests}

CLASSIFIER = 'Classifier'
TYPE = 'Type/Parameter'

accuracies = pd.DataFrame(columns=[CLASSIFIER, TYPE, 'Accuracy'])
sens = pd.DataFrame(columns=[CLASSIFIER, TYPE, 'Sensibility', 'Specificity'])

for model_type in classifiers:
    for specific in classifiers[model_type]:
        clf, parameter = specific
        res = classifier_statistics(clf, X_train, X_test, y_train, y_test)

        accuracy = res['accuracy']
        sensibility = res['sensibility']
        specificity = res['specificity']

        

        accuracies = accuracies.append({CLASSIFIER: model_type, TYPE: parameter, 'Accuracy': accuracy}, ignore_index=True)
        sens = sens.append({CLASSIFIER: model_type, TYPE: parameter, 'Sensibility': sensibility, 'Specificity': specificity}, ignore_index=True)

ax = sns.barplot(x=CLASSIFIER, y='Accuracy', hue=TYPE, data=accuracies)
plt.savefig('images/initial_accuracies.pdf')
plt.clf()
