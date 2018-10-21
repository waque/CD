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
        classifier_statistics
)

sns.set()
sns.set_style("whitegrid")
sns.despine()

CLASS = 'class'

"""

Problem description:
    Minimize the costs of repairing APS.
    Try to predict when a failure is going to occur.
    Predicting a failure when it's not cost: 10
    Missing a failure cost: 500
    (this is available in the txt that comes with the dataset)
"""

aps_train = pd.read_csv('./aps_failure_training_set.csv',
                        skiprows=20,keep_default_na=False)


aps_test = pd.read_csv('./aps_failure_test_set.csv',
                        skiprows=20,keep_default_na=False)

"""

Problem:
    Missing values are represented as 'na' need to replace those with nulls
    
Explore:
    Different strategies of replacing the nans values and comparing the results

"""

aps_train.replace('na', np.nan, inplace=True)
aps_test.replace('na', np.nan, inplace=True)

"""

Problem:
    Data balancing, the classes distribution is not balanced
    this is normal since this is a failure thing and it's normal
    that it doesn't fail as much as it works
"""

print(aps_train[CLASS].value_counts())
print(aps_train[CLASS].value_counts())

"""
train_hist = sns.countplot(x=CLASS, data=aps_train)
train_hist.set(xlabel='Failure', ylabel='Count')
plt.savefig('unbalanced_train.pdf')
plt.clf()

test_hist = sns.countplot(x=CLASS, data=aps_test)
test_hist.set(xlabel='Failure', ylabel='Count')
plt.savefig('unbalanced_test.pdf')
plt.clf()
"""

X_train, y_train = split_dataset(aps_train, CLASS)
X_test, y_test = split_dataset(aps_train, CLASS)

y_train = y_train.map({'pos': 1, 'neg': 0})
y_test = y_test.map({'pos': 1, 'neg': 0})

"""

Good thing:
    no class attributes, no need for dummy transformations
    only on the class itself

"""

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')


"""
Data exploration:
    There are 170 columns, all of them are numeric values.

"""
    
    