import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
from utils_cd import *
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_selection import SelectKBest
from imblearn.over_sampling import SMOTE

CLASS = 'consensus'

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

#QUERO VER SER DÁ ERRO AQUI
sns.boxplot(x=super_table['os_area'])

from scipy import stats
import numpy as np
z = np.abs(stats.zscore(super_table))
print(z)

threshold = 3
print(np.where(z > 3))
