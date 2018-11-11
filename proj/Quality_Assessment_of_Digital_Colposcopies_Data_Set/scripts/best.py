import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from utils_cd import (
        split_dataset,
        standard_deviation,
        plot_comparison_results,
        impute_values,
        plot_results,
        plot_param_improv,
        train_test_split,
        plot_results_from_csv,
        aps_classifier_statistics,
        classifier_statistics,
        cv_classifier_statistics
)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

pd.options.display.max_columns = None
sns.set(style='darkgrid')
CLASS = 'consensus'

def balance_dataset(dataset, classe):
    y = dataset[classe]
    X = dataset.drop(columns=[classe])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sm = SMOTE(random_state=2)

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
    
    

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res


def getKBest(X, y, score_func=f_classif, k=10):
    k_best = SelectKBest(score_func=score_func, k=10).fit(X, y)

    idxs = k_best.get_support(indices=True)
    X = X.iloc[:,idxs]
    return X

clf1 = BernoulliNB()
clf2 = DecisionTreeClassifier()
clf3 = KNeighborsClassifier()
clf4 = RandomForestClassifier(n_estimators=10)
eclf1 = VotingClassifier(estimators=[('nb', clf1), ('bt', clf2), ('knn', clf3), ('rf', clf4)], voting='hard')

base_clfs = [BernoulliNB(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(n_estimators=10), eclf1]

green_data = pd.read_csv('../green.csv')
hinselmann_data = pd.read_csv('../hinselmann.csv')
schiller_data = pd.read_csv('../schiller.csv')

data = [[green_data,'green_data'], [hinselmann_data,'hinselmann_data'], [schiller_data,'schiller_data']]

green_data['hinselmann']=0
green_data['schiller']=0
hinselmann_data['hinselmann']=1
hinselmann_data['schiller']=0
schiller_data['hinselmann']=0
schiller_data['schiller']=1

super_table = green_data.append(hinselmann_data)
super_table = super_table.append(schiller_data)

X, y = split_dataset(super_table, CLASS)
X = getKBest(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

results = {}

for clf in base_clfs:
    clf_name = type(clf).__name__
    stats = classifier_statistics(clf, X_train_res, X_test, y_train_res, y_test)
    results[clf_name] = stats

measures = {}
i = 0
for clf in results:
    clf_res = results[clf]
    measures[i] = {'Classifier': clf, 'Measure': 'Accuracy', 'Value': clf_res['accuracy']}
    i += 1
    measures[i] = {'Classifier': clf, 'Measure': 'Sensibility', 'Value': clf_res['sensibility']}
    i += 1
    measures[i] = {'Classifier': clf, 'Measure': 'Specificity', 'Value': clf_res['specificity']}
    i += 1
        
measures = pd.DataFrame.from_dict(measures, "index")
measures.to_csv('../plot_data/{}.csv'.format('super_dateset_balanced_KBest_Voting'))
