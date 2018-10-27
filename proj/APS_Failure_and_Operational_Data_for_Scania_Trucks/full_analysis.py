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
        standard_deviation,
        impute_values,
        plot_results
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

sns.set(style="whitegrid")

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

X_train, y_train = split_dataset(orig_train, CLASS)
X_test, y_test = split_dataset(orig_test, CLASS)
#X_train = imp.fit_transform(X_train)
#X_test = imp.fit_transform(X_test)

y_train = y_train.map({'pos': 1, 'neg': 0})
y_test = y_test.map({'pos': 1, 'neg': 0})

"""
neighbors = [3, 5, 10]
estimators = [25, 50, 100]

naive_bayes = [(GaussianNB(), 'Gaussian'), (MultinomialNB(), 'Multinomial'), (BernoulliNB(), 'Bernoulli')]
knns = [(KNeighborsClassifier(n_neighbors=x), 'K Nearest Neighbors {}'.format(x)) for x in neighbors]
random_forests = [(RandomForestClassifier(n_estimators=x), 'Random Forest {}'.format(x)) for x in estimators]

classifiers = {'Naive Bayes': naive_bayes, 'K Nearest Neighbors': knns, 'Random Forest': random_forests}
CLASSIFIER = 'Classifier'

measures_dict = {}
i = 0
for model_type in classifiers:
    for specific in classifiers[model_type]:
        clf, parameter = specific
        res = classifier_statistics(clf, X_train, X_test, y_train, y_test)

        accuracy = res['accuracy']
        sensibility = res['sensibility']
        specificity = res['specificity']
        measures_dict[i] = {CLASSIFIER: parameter, 'Measure': 'Accuracy', 'Value': accuracy}
        i += 1
        measures_dict[i] = {CLASSIFIER: parameter, 'Measure': 'Sensibility', 'Value': sensibility}
        i += 1
        measures_dict[i] = {CLASSIFIER: parameter, 'Measure': 'Specificity', 'Value': specificity}
        i += 1


measures = pd.DataFrame.from_dict(measures_dict, "index")
measures.to_csv('plot_data/initial_results.csv')
plt.figure(figsize=(22,6))
ax = sns.barplot(x=CLASSIFIER, y='Value', hue='Measure', data=measures)
plt.savefig('images/initial_results.pdf')
plt.clf()
"""

"""
Maybe we should plot this with a roc curve

Most of the classifiers didn't present good sensibility
values, which in the domain of the problem is bad. We wish
to detect the maximum number of true positives possible
while maintaining a not too high value of false positives. 
We will be using the bernoylli naive bayes classifier
to test different preprocessing techniques as it
presented the highest sensibility while still maintaining a 
good value for the specificity.

"""

rows_na = orig_train.shape[0] - orig_train.dropna().shape[0]
percentage = rows_na / orig_train.shape[0]
print('Rows with missing values: {}'.format(orig_train.shape[0] - orig_train.dropna().shape[0]))
print('Percentage of rows with missing values: {}'.format(percentage))

"""

The dataset has 99.015% of the instances containing missing
values. We will have to find out the best way to treat this missing
values to obtain the best results. We already have tested replacing
missing values with 0. The other approaches we will be testing
are replacing with the median, mean and most_frequent
We will also be testing removing attributes with too many missing values
and see if we obtain better results.

"""
"""
X_train_orig, X_test_orig = impute_values(X_train, X_test, "constant", constant=0)
X_train_mean, X_test_mean = impute_values(X_train, X_test, "mean")
X_train_median, X_test_median = impute_values(X_train, X_test, "median")
X_train_mfrequent, X_test_mfrequent = impute_values(X_train, X_test, "most_frequent")

X_data = {'Original': (X_train_orig, X_test_orig), 'Mean': (X_train_mean, X_test_mean), 'Median': (X_train_median, X_test_median), 'Most Frequent': (X_train_mfrequent, X_test_mfrequent)}
clf = BernoulliNB()
plot_results(clf, X_data, y_train, y_test, filename='missing_imputation')
"""
"""

After analysis of the results we can see that
the best results are achieved with the original imputation.
Even though the accuracy increases in the others the sensibility
decreases. This is because of the unbalanced nature of the data,
we could test this again after balancing the data to see if the results change.
We will continue the preprocessing using the original missing values imputation.

"""

"""
Still in the process of data cleaning we analyzed the dataset
to check for attributes that add a high percentage of missing values
or that didn't add any value for our model.
We considered a treshold of 60% and 80%v of missing values as a criteria
to remove columns from the dataset. We also found a column which
the value didn't vary and so it didn't add any information for our model
so we removed it aswell.
"""

"""

columns_to_remove = ['br_000', 'bq_000', 'bp_000', 'bo_000', 'ab_000', 'cr_000', 'bn_000', 'bm_000', 'cd_000']
columns_to_remove2 = ['br_000', 'bq_000', 'cd_000']

X_train_drop = X_train.drop(columns=columns_to_remove)
X_test_drop = X_test.drop(columns=columns_to_remove)
X_train_drop, X_test_drop = impute_values(X_train_drop, X_test_drop, "constant", constant=0)

X_train_drop2 = X_train.drop(columns=columns_to_remove2)
X_test_drop2 = X_test.drop(columns=columns_to_remove2)
X_train_drop2, X_test_drop2 = impute_values(X_train_drop2, X_test_drop2, "constant", constant=0)


X_data = {'Original': (X_train_orig, X_test_orig), 'Removed Columns 60%': (X_train_drop, X_test_drop), 'Removed Columns 80%': (X_train_drop2, X_test_drop2)}
clf = BernoulliNB()
results = plot_results(clf, X_data, y_train, y_test, filename='removed_columns')
"""


"""
After analyzing the values we can see that removing this columns
wasn't benifical because we lost sensibility which is the most important metric for our problem
"""