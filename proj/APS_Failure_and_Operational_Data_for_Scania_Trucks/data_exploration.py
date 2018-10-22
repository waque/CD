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
#plt.savefig('unbalanced_train.pdf')
plt.clf()

test_hist = sns.countplot(x=CLASS, data=aps_test)
test_hist.set(xlabel='Failure', ylabel='Count')
#plt.savefig('unbalanced_test.pdf')
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
    talk about training and test dataset instances
    There is no information about the attributes real meaning
    because they are anonymized for proprietary issues. We can't inffer attributes
    because of this.
    There are attributes that correspond to intervals, "bins",
    discretize (?) this attributes.
"""

"""
Problem:
    Missing data.
"""

print('Number of rows after removing training missing values: {}'.format(aps_train.dropna().shape[0]))
print('Number of rows after removing test missing values: {}'.format(aps_test.dropna().shape[0]))

"""

the number of instances available after removing the missing values is 
very low comparing to the original dataset.
Try to explore the dataset to remove columns that have to many missing values or 
values that don't vary accross instances

I think it makes more sense to work with both datasets concatenated, so we can see 
the overall thing.

"""

aps = pd.concat([aps_train, aps_test])
X, y = split_dataset(aps, CLASS)
X = X.astype('float64')

num_instances = X.shape[0]

for col in aps:
    print('Analyzing col {}'.format(col))
    print()
    std_dev = standard_deviation(aps[col])
    num_missing = aps[col].isna().count()
    