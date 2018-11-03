import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from utils_cd import (
        split_dataset,
        standard_deviation,
        plot_comparison_results,
        impute_values,
        plot_results,
        plot_param_improv,
        train_test_split,
        classifier_statistics
)
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import Normalizer, StandardScaler
from imblearn.over_sampling import SMOTE

green_data = pd.read_csv('./green.csv')
hinselmann_data = pd.read_csv('./hinselmann.csv')
schiller_data = pd.read_csv('./schiller.csv')

def balance_dataset(dataset, classe):
    y = dataset[classe]
    X = dataset.drop(columns=[classe])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sm = SMOTE(random_state=2)

    X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
    
    

    return X_train, X_test, y_train, y_test, X_train_res, y_train_res


naive_cls = GaussianNB()

green_data['hinselmann']=0
green_data['schiller']=0
hinselmann_data['hinselmann']=1
hinselmann_data['schiller']=0
schiller_data['hinselmann']=0
schiller_data['schiller']=1

super_table = green_data.append(hinselmann_data)
super_table = super_table.append(schiller_data)

sns.set()
sns.set_style("whitegrid")
sns.despine()

#ax = sns.countplot(x="consensus", data=super_table)
#plt.savefig('super_table_class.pdf')
#plt.clf()

X_train, X_test, y_train, y_test, X_train_res, y_train_res = balance_dataset(super_table, 'consensus')

#ax = sns.countplot(x="consensus", data=super_table)
#plt.savefig('super_table_bal_class.pdf')
#plt.clf()

"""
need to transform numpy dataset to pandas!
"""