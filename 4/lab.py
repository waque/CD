import sys
sys.path.append('..')
from utils_cd import (
        split_dataset_transformed,
        cross_val,
        print_dict,
        split_train_test,
        classifier_statistics
)

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer

def exercise1():
    breast_cancer = pd.read_csv('breast_cancer.csv')
    
    X, y = split_dataset_transformed(breast_cancer, 'Class', ['?'])
    
    # 1.a
    clf = DecisionTreeClassifier()
    cross_val_stats = cross_val(clf, X, y)
    print_dict(cross_val_stats)
    
    
    # 1.b
    
    min_samples = np.arange(2, 11)
    
    for samples in min_samples:
        print('Experimenting with {} samples'.format(samples))
    
        clf = DecisionTreeClassifier(min_samples_split=samples)
        cross_val_stats = cross_val(clf, X, y)
        print_dict(cross_val_stats)
        
    # 1.c   
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    export_graphviz(clf, out_file='unpruned.dot')
    
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    export_graphviz(clf, out_file='pruned.dot')


def exercise2():
    breast_cancer = pd.read_csv('breast_cancer.csv')
        
    X, y = split_dataset_transformed(breast_cancer, 'Class', ['?'])
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # 2.a
    
    clf = RandomForestClassifier()
    clf_stats = classifier_statistics(clf, X_train, X_test, y_train, y_test)
    print_dict(clf_stats, ['predicted'])
    
    # 2.b
    
    numb_trees = np.arange(10, 201, step=10)
    
    for trees in numb_trees:
        print('Experimenting with {} number of trees'.format(trees))
    
        clf = RandomForestClassifier(n_estimators=trees)
        clf_stats = classifier_statistics(clf, X_train, X_test, y_train, y_test)
        print_dict(clf_stats, ['predicted'])
        
    # 2.c
        
    
    depths = np.arange(5, 20)
    
    for dep in depths:
        print('Experimenting with {} depth'.format(dep))
    
        clf = RandomForestClassifier(max_depth=dep)
        clf_stats = classifier_statistics(clf, X_train, X_test, y_train, y_test)
        print_dict(clf_stats, ['predicted'])
    

def exercise3():
    
    credit = pd.read_csv('credit.csv')
            
    X, y = split_dataset_transformed(credit, 'class')
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    min_samples = np.arange(2, 11)
    
    for samples in min_samples:
        print('Experimenting with {} number of instances to split'.format(samples))
        print('Train data')
        
        clf = DecisionTreeClassifier(min_samples_split=samples)
        clf_stats = classifier_statistics(clf, X_train, X_train, y_train, y_train)
        print_dict(clf_stats, ['predicted'])
    
    
        print('Test data')
        
        clf = DecisionTreeClassifier(min_samples_split=samples)
        clf_stats = classifier_statistics(clf, X_train, X_test, y_train, y_test)
        print_dict(clf_stats, ['predicted'])
        
        print()
        print()
    
    
    numb_trees = np.arange(10, 201, step=10)
        
    for trees in numb_trees:
        print('Experimenting with {} number of trees'.format(trees))
        print('Train data')
    
        clf = RandomForestClassifier(n_estimators=trees)
        clf_stats = classifier_statistics(clf, X_train, X_train, y_train, y_train)
        print_dict(clf_stats, ['predicted'])
        
        print('Test data')
        
        clf = RandomForestClassifier(n_estimators=trees)
        clf_stats = classifier_statistics(clf, X_train, X_test, y_train, y_test)
        print_dict(clf_stats, ['predicted'])
    
        print()
        print()



diabetes = pd.read_csv('diabetes.csv')
X, y = split_dataset_transformed(diabetes, 'class')
X_train, X_test, y_train, y_test = split_train_test(X, y)
    

# 3.a

enc = KBinsDiscretizer(n_bins=10, encode='onehot')
X_binned = enc.fit_transform(X)
X_binned = X_binned.toarray()
X_binned_train, X_binned_test, y_binned_train, y_binned_test = split_train_test(X_binned, y)

print('With discretization')

clf1 = RandomForestClassifier()
clf_stats = classifier_statistics(clf1, X_binned_train, X_binned_test, y_binned_train, y_binned_test)
print_dict(clf_stats, ['predicted'])

print('Without discretization')

clf2 = RandomForestClassifier()
clf_stats = classifier_statistics(clf2, X_train, X_test, y_train, y_test)
print_dict(clf_stats, ['predicted'])

