#because of the stupid MacOS X
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import seaborn as sns
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
from scipy import stats


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


#sns.boxplot(x=super_table['os_area'])
#plt.savefig('test.pdf')
#plt.clf()
"""
The Z-score is the signed number of standard deviations by which the value
of an observation or data point is above the mean value of what is being 
observed or measured.
"""
z = np.abs(stats.zscore(super_table))
threshold = 3

#fig, ax = plt.subplots(figsize=(16,8))
#ax.scatter(super_table['cervix_area'], super_table['os_area'])
#plt.show()

"""
The interquartile range (IQR), also called the midspread or middle 50%, 
or technically H-spread, is a measure of statistical dispersion, being 
equal to the difference between 75th and 25th percentiles, or between 
upper and lower quartiles, IQR = Q3 − Q1.

In other words, the IQR is the first quartile subtracted from the third 
quartile; these quartiles can be clearly seen on a box plot on the data.

It is a measure of the dispersion similar to standard deviation or variance, 
but is much more robust against outliers.
"""


#print(super_table.head())

Q1 = super_table.quantile(0.25)
Q3 = super_table.quantile(0.75)
IQR = Q3 - Q1
#print(IQR)

#print(super_table < (Q1 - 1.5 * IQR)) |(super_table > (Q3 + 1.5 * IQR))

super_table_1 = super_table[(z < 3).all(axis=1)]
#print(super_table.shape)
#print(super_table_1.shape)

"""
A remover os outliers, são removidas 74 rows. Passando de 287 rows para 213 rows.
Uma diminuição de 25% das rows -> not very good
"""

super_table_out = super_table[~((super_table < (Q1 - 1.5 * IQR)) |(super_table > (Q3 + 1.5 * IQR))).any(axis=1)]
#print(super_table_out.shape)

"""
o IQR é péssimo! Remove 229 rows!
Remove 80% das rows
"""

"""
Vamos agora testar a a super table sem os outliers usando o método z-score
"""


clf1 = BernoulliNB()
clf2 = DecisionTreeClassifier()
clf3 = KNeighborsClassifier()
clf4 = RandomForestClassifier(n_estimators=10)
eclf1 = VotingClassifier(estimators=[('nb', clf1), ('bt', clf2), ('knn', clf3), ('rf', clf4)], voting='hard')

base_clfs = [BernoulliNB(), DecisionTreeClassifier(), KNeighborsClassifier(), RandomForestClassifier(n_estimators=10), eclf1]

X, y = split_dataset(super_table_1, CLASS)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#concatenar X e y de traino para remover outliers
#train = pd.concat([X_train, y_train], axis=1)
#train = train[(z < 3).all(axis=1)]
#X_train = train.iloc[:, train.columns != CLASS]
#y_train = train[CLASS]

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
measures.to_csv('../plot_data/{}.csv'.format('super_dateset_balanced_outliers'))
