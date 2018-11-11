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
        classifier_statistics
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA

def balance_dataset(dataset, classe):
    y = dataset[classe]
    X = dataset.drop(columns=[classe])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sm = SMOTE(random_state=2)

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
    
    

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res

def normalize(X_train, X_test):
    normalizer = Normalizer().fit(X_train)

    X_train_norm = normalizer.transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    
    return X_train_norm, X_test_norm


naive_cls = GaussianNB()

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

X_train_super, X_test_super, y_train_super, y_test_super, X_train_res_super, y_train_res_super = balance_dataset(super_table, 'consensus')

X_train_norm, X_test_norm = normalize(X_train_super, X_test_super)

res_super_unbal = classifier_statistics(naive_cls, X_train_norm, X_test_norm, y_train_super, y_test_super)
"""
normalization com super table unbalenced -> resultados muito piores
accuracy': 0.8045977011494253, 'sensibility': 0.9508196721311475, 'specificity': 0.46153846153846156
"""

X_train_norm, X_test_norm = normalize(X_train_res_super, X_test_super)
res_super_bal = classifier_statistics(naive_cls, X_train_norm, X_test_norm, y_train_res_super, y_test_super)
print(res_super_bal)

"""
normalization com super table unbalenced -> resultados muito piores
'accuracy': 0.8160919540229885, 'sensibility': 0.9180327868852459, 'specificity': 0.5769230769230769
"""

