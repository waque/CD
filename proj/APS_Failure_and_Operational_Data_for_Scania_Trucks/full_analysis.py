import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
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
import scipy.cluster.hierarchy as hac
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from imblearn.over_sampling import SMOTE

def score(conf_matrix):
    cost1 = 10
    cost2 = 500
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]

    total_cost = cost1*fp + cost2*fn

    print('Total cost aachived: {}'.format(total_cost))

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
X_train, X_test= X_train.fillna(0), X_test.fillna(0)
y_train = y_train.map({'pos': 1, 'neg': 0})
y_test = y_test.map({'pos': 1, 'neg': 0})
balanced = False

print('Balanced data: {}'.format(collections.Counter(y_train)))

def get_data(X_train=X_train, X_test=X_test):
    global y_train
    global balanced
    high_corr = get_high_correlated_cols()
    X_train = X_train.drop(columns=high_corr)
    X_test = X_test.drop(columns=high_corr)
    if not balanced:
        sm = SMOTE(random_state=12, ratio = 1.0)
        X_train, y_train = sm.fit_sample(X_train, y_train)
        balanced = True    
    
    return X_train, X_test
#X_train = imp.fit_transform(X_train)
#X_test = imp.fit_transform(X_test)
def compare_baseline_clf():
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

            conf_matrix = res['confusion_matrix']
            score(conf_matrix)

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
    #plt.savefig('images/initial_results.pdf')
    plt.clf()


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
def print_missing_val_info():
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
and see if we obtain better results. This can be because of different
distributions between training and testing set.

"""
def compare_imputations():

    X_train_orig, X_test_orig = impute_values(X_train, X_test, "constant", constant=0)
    X_train_mean, X_test_mean = impute_values(X_train, X_test, "mean")
    X_train_median, X_test_median = impute_values(X_train, X_test, "median")
    X_train_mfrequent, X_test_mfrequent = impute_values(X_train, X_test, "most_frequent")

    X_data = {'Original': (X_train_orig, X_test_orig), 'Mean': (X_train_mean, X_test_mean), 'Median': (X_train_median, X_test_median), 'Most Frequent': (X_train_mfrequent, X_test_mfrequent)}
    clf = BernoulliNB()
    plot_results(clf, X_data, y_train, y_test, filename='missing_imputation')


"""

After analysis of the results we can see that
the best results are achieved with the original imputation.
Even though the accuracy increases in the others the sensibility
decreases. This is because of the unbalanced nature of the data,
we could test this again after balancing the data to see if the results change.
We will continue the preprocessing using the original missing values imputation.

"""

"""
Correlation between attributes
"""


def dendrogram(df):    
    # Do correlation matrix
    corr_matrix = df.corr()

    # Do the clustering
    Z = hac.linkage(corr_matrix, 'single')

    # Plot dendogram
    fig, ax = plt.subplots(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    groups = hac.dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8., # font size for the x axis labels
        color_threshold = 0#,
        #truncate_mode='lastp',
        #p=30
    )

    labels_dict = pd.DataFrame(df.columns).to_dict()[0]
    actual_labels = [item.get_text() for item in ax.get_xticklabels()]
    new_labels = [labels_dict[int(i)] for i in actual_labels]
    ax.set_xticklabels(new_labels)
    plt.tight_layout()
    plt.savefig('images/dendrogram.pdf')
    plt.clf()



def attribute_corr(df, top):
        
    def get_redundant_pairs(df):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df, n=5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        return au_corr[0:n]

    
    return get_top_abs_correlations(df, top)

def get_high_correlated_cols():
    top = attribute_corr(X_train, 15)
    top = top[top > 0.999]
    cols = []

    for attr1, attr2 in top.index.to_series().values:
        #print('Attribute {} and {} are highly correlated'.format(attr1, attr2))
        missing1 = X_train[attr1].isna().sum()
        missing2 = X_train[attr2].isna().sum()

        #print ('Missing values: {} - {}, {} - {} '.format(attr1, missing1, attr2, missing2))

        if missing1 > missing2:
            cols.append(attr1)
        else:
            cols.append(attr2)

    return cols
            
"""
PCA:
attributes 
bb_000  bv_000    1.000000
ah_000  bg_000    1.000000
bv_000  cq_000    1.000000
bb_000  cq_000    1.000000
aa_000  bt_000    1.000000
bu_000  cq_000    1.000000
bv_000    1.000000
bb_000  bu_000    1.000000
cf_000  co_000    1.000000
ad_000  cf_000    1.000000
co_000    1.000000

have high correlation, explore the ammount of missing values
between them
['bv_000',
 'ah_000',
 'cq_000',
 'cq_000',
 'bt_000',
 'cq_000',
 'bv_000',
 'bu_000',
 'co_000',
 'cf_000',
 'co_000']
"""

def drop_corr_columns():
    X_train_orig, X_test_orig = get_data(X_train, X_test)

    high_corr = get_high_correlated_cols()
    X_train_drop = X_train.drop(columns=high_corr)
    X_test_drop = X_test.drop(columns=high_corr)
    X_train_drop, X_test_drop = impute_values(X_train_drop, X_test_drop, "constant", constant=0)

    X_data = {'Original': (X_train_orig, X_test_orig), 'Dropped high correlated cols': (X_train_drop, X_test_drop)}
    clf = BernoulliNB()
    results = plot_results(clf, X_data, y_train, y_test, filename='high_correlated')

"""
Still in the process of data cleaning we analyzed the dataset
to check for attributes that add a high percentage of missing values
or that didn't add any value for our model.
We considered a treshold of 60% and 80%v of missing values as a criteria
to remove columns from the dataset. We also found a column which
the value didn't vary and so it didn't add any information for our model
so we removed it aswell.
"""

def column_removal():
    X_train_orig, X_test_orig = get_data()

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
After analyzing the values we can see that removing this columns
wasn't beneficial because we lost sensibility which is the most important metric for our problem
"""

"""

Data transformation:
Normalization of attributes
There were no changes with normalization using the bernoullin
other classifiers change

"""

def normalize_attributes():
    X_train_orig, X_test_orig = get_data()
    normalizer = preprocessing.Normalizer().fit(X_train_orig)

    X_train_norm = normalizer.transform(X_train_orig)
    X_test_norm = normalizer.transform(X_test_orig)

    X_data = {'Original': (X_train_orig, X_test_orig), 'Normalized': (X_train_norm, X_test_norm)}
    clf = RandomForestClassifier(n_estimators=100)
    results = plot_results(clf, X_data, y_train, y_test, filename='normalized_after_balance_rf')

    conf_matrix = results['Original']['confusion_matrix']
    score(conf_matrix)
    conf_matrix = results['Normalized']['confusion_matrix']
    score(conf_matrix)
    

"""

standardize distributions

"""

def standardize():
    X_train_orig, X_test_orig = get_data()
    scaler = preprocessing.StandardScaler().fit(X_train_orig)

    X_train_std = scaler.transform(X_train_orig)
    X_test_std = scaler.transform(X_test_orig)

    X_data = {'Original': (X_train_orig, X_test_orig), 'Standardized': (X_train_std, X_test_std)}
    clf = RandomForestClassifier(n_estimators=100)
    results = plot_results(clf, X_data, y_train, y_test, filename='standardized_after_balance_rf')

    conf_matrix = results['Original']['confusion_matrix']
    score(conf_matrix)
    conf_matrix = results['Standardized']['confusion_matrix']
    score(conf_matrix)

    print(results)
    

"""

balancing the datings

"""
