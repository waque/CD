import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix


def print_dict(dic, excluded_keys=[]):
    for key in dic:
        if key not in excluded_keys:
            print('{}: {}'.format( key, dic[key]))

def sensibility(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tp / (tp + fn)

def specificity(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tn / (tn + fp)

def standard_deviation(a):
    return np.std(a)

def mean(a):
    return np.mean(a)

def sad(a, b):
    return np.sum(abs(a - b))

def ssd(a, b):
    return np.sum(np.square(a - b))

def correlation(a, b):
    return np.corrcoef(np.array((a, b)))[0, 1]

def transform_X(X):
    table = X
    for col in table:
        if table[col].dtype == np.object:
            col_dummies = pd.get_dummies(table[col], prefix=col, drop_first=True)
            table = pd.concat([table, col_dummies], axis=1)
            table = table.drop(columns=col)
    return table

def transform_y(y):
    return y.astype('category').cat.codes

def split_dataset(dataset, y_name, missing_values=None):
    if missing_values:
        for value in missing_values:
            dataset = dataset[~dataset.eq(value).any(1)]
    
    X = dataset.iloc[:, dataset.columns != y_name]
    y = dataset[y_name]
    
    return X, y

def split_dataset_transformed(dataset, y_name, missing_values=None):
    X, y = split_dataset(dataset, y_name, missing_values)
    return transform_X(X), transform_y(y)

def split_train_test(X, y, test_size=0.3):
    # returns X_train, X_test, y_train, y_test
    
    return train_test_split(X, y, test_size=test_size, random_state=42)

def classifier_statistics(clf, X_train, X_test, y_train, y_test):
    res = {}
    
    clf.fit(X_train, y_train)
    
    predicted = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predicted)
    acc_score = accuracy_score(y_test, predicted)
    sens = sensibility(conf_matrix)
    spec = specificity(conf_matrix)
    
    res['predicted'] = predicted
    res['accuracy'] = acc_score
    res['confusion_matrix'] = conf_matrix
    res['sensibility'] = sens
    res['specificity'] = spec
    
    return res

def cross_val(clf, X, y, folds=3):
    scores = cross_val_score(clf, X, y, cv=folds)
    
    res = {}
    
    res['mean_accuracy'] = mean(scores)
    res['standard_deviation'] = standard_deviation(scores)
    
    return res
    