import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from imblearn.over_sampling import SMOTE
sys.path.append('..')
from utils_cd import (
        split_dataset_transformed,
        cross_val,
        print_dict,
        split_dataset,
        classifier_statistics,
        split_train_test
)


sns.set()
sns.set_style("whitegrid")
sns.despine()



"""
Problem description:

"""

green_data = pd.read_csv('./green.csv')
hinselmann_data = pd.read_csv('./hinselmann.csv')
schiller_data = pd.read_csv('./schiller.csv')

#X_green, y_green = split_dataset_transformed(green_data, 'consensus')
#X_train_green, X_test_green, y_train_green, y_test_green = split_train_test(X_green, y_green, test_size=0.3)

#sm = SMOTE(random_state=12, ratio = 1.0)

#X_train_green_res, y_train_green_res = sm.fit_sample(X_train_green, y_train_green)

#print("Before OverSampling, counts of label '1': {}".format(sum(y_train_green==1)))
#print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_green==0)))

#print("After OverSampling, counts of label '1': {}".format(sum(y_train_green_res==1)))
#print("After OverSampling, counts of label '0': {}".format(sum(y_train_green_res==0)))


balance_dataset(green_data, 'consensus')
balance_dataset(hinselmann_data, 'consensus')
balance_dataset(schiller_data, 'consensus')

"""
ax = sns.countplot(x="consensus", data=green_data)
plt.savefig('green_consensus_balance.pdf')
plt.clf()

ax = sns.countplot(x="consensus", data=hinselmann_data)
plt.savefig('hinselmann_consensus_balance.pdf')
plt.clf()

ax = sns.countplot(x="consensus", data=schiller_data)
plt.savefig('schiller_consensus_balance.pdf')
plt.clf()

"""

"""
Data exploration:
    There are 69 columns, all of them are numeric values.
    hinselmann_data class not balanced 
    

"""
    
    