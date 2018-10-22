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



"""
Problem description:

"""

green_data = pd.read_csv('./green.csv')
hinselmann_data = pd.read_csv('./hinselmann.csv')
schiller_data = pd.read_csv('./schiller.csv')


#X_train_green, X_test_green, y_train_green, y_test_green = split_dataset_transformed(green_data, 'consensus')
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
    
    