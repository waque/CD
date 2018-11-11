import pandas as pd
import numpy as np
import seaborn as sns
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc


def aps_score(conf_matrix):
    cost1 = 10
    cost2 = 500
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]

    return cost1*fp + cost2*fn

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
    a = np.nan_to_num(a)
    return np.mean(a)

def sad(a, b):
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
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


def cv_classifier_statistics(clf, X, y, k=10):
    res = {'predicted': [], 'accuracy': [], 'confusion_matrix': [], 'sensibility': [], 'specificity': [], 'auc': [], 'fpr': np.linspace(0, 1, 100), 'tpr': []}
    clf = clone(clf)

    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        stats = classifier_statistics(clf, X_train, X_test, y_train, y_test)

        fpr, tpr, _ = roc_curve(y[test_index], stats['predicted'])
        roc_auc = auc(fpr, tpr)
        res['tpr'].append(interp(res['fpr'], fpr, tpr))
        res['auc'].append(roc_auc)

        for stat in stats:
            res[stat].append(stats[stat])
        
    res_cp = res.copy()
    for key in res_cp:
        if key != 'confusion_matrix' and key != 'predicted':
            res['{}_mean'.format(key)] = (mean(res[key]), standard_deviation(res[key]))
    
    return res


def aps_classifier_statistics(clf, X_train, X_test, y_train, y_test):
    res = classifier_statistics(clf, X_train, X_test, y_train, y_test)
    res['score'] = aps_score(res['confusion_matrix'])

    return res

def classifier_statistics(clf, X_train, X_test, y_train, y_test):
    res = {}

    clf = clone(clf)
    
    clf.fit(X_train, y_train)
    
    predicted = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predicted, labels=[0.0, 1.0])
    acc_score = accuracy_score(y_test, predicted)
    sens = sensibility(conf_matrix)
    spec = specificity(conf_matrix)
    
    res['predicted'] = predicted
    res['accuracy'] = acc_score
    res['confusion_matrix'] = conf_matrix
    res['sensibility'] = sens
    res['specificity'] = spec
    res['clf'] = clf
    
    return res

def cross_val(clf, X, y, folds=3):
    scores = cross_val_score(clf, X, y, cv=folds)
    
    res = {}
    
    res['mean_accuracy'] = mean(scores)
    res['standard_deviation'] = standard_deviation(scores)
    
    return res
    
def pprint(res):
    print("\nthis classifier got the following results: \n")
    print("accuracy: " + str(res['accuracy']) + "\n")
    print("sensibility: " + str(res['sensibility']) + "\n")
    print("specificity: " + str(res['specificity']) + "\n")

def export_file(res, data_name, classifier_name, extra_info):
    f = open("dataset_info.txt", "a")
    f.write("the classifier " + classifier_name + " using " + extra_info + " with " + data_name + " got the following results:" + "\n")
    f.write("accuracy: " + str(res['accuracy']) + "\n")
    f.write("sensibility: " + str(res['sensibility']) + "\n")
    f.write("specificity: " + str(res['specificity']) + "\n\n")

def impute_values(X_train, X_test, strategy, missing_values=np.nan, constant=None):
    if not constant:
        imp = SimpleImputer(missing_values=missing_values, strategy=strategy)
    else:
        imp = SimpleImputer(strategy="constant", fill_value=constant)
    
    imp = imp.fit(X_train)

    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)
    return X_train, X_test

def plot_comparison_results(clf_list, X_data, y_train, y_test, technique='Technique', filename='result', style='darkgrid', figsize=(16,6)):
    sns.set(style=style)
    measures_dict = {}
    i = 0
    results = {}
    for var in X_data:
        X_train, X_test = X_data[var]
        results[var] = {}
        for clf in clf_list:
            clf = clone(clf)
            res = classifier_statistics(clf, X_train, X_test, y_train, y_test)
            score = aps_score(res['confusion_matrix'])
            res['score'] = score
            results[var][type(clf).__name__] = res
            measures_dict[i] = {technique: var, 'Classifier': type(clf).__name__, 'Price': score}
            i += 1
    
    measures = pd.DataFrame.from_dict(measures_dict, "index")
    measures.to_csv('plot_data/{}.csv'.format(filename))
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Classifier', y='Price', hue=technique, data=measures)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.3f}'.format(float(p.get_height())), 
            fontsize=10, color='black', ha='center', va='bottom')
    
    plt.savefig('images/{}.pdf'.format(filename))
    plt.clf()

    return results

def plot_results_from_csv(filename='filename', technique='Technique', figsize=(16, 6)):
    measures = pd.read_csv('plot_data/{}.csv'.format(filename))
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Classifier', y='Price', hue=technique, data=measures)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.3f}'.format(float(p.get_height())), 
            fontsize=8, color='black', ha='center', va='bottom')
    
    plt.savefig('images/{}.pdf'.format(filename))
    plt.clf()

    return measures

def plot_results(clf_list, X_data, y_train, y_test, technique='Technique', filename='result', style='darkgrid', figsize=(16,6)):
    sns.set(style=style)
    measures_dict = {}
    i = 0
    results = {}
    for clf in clf_list:
        clf = clone(clf)
        results[type(clf).__name__] = {}
        for var in X_data[type(clf).__name__]:
            X_train, X_test = X_data[type(clf).__name__][var]
            curr_y_train = y_train
            if isinstance(y_train, dict):
                curr_y_train = y_train[type(clf).__name__][var]
            res = classifier_statistics(clf, X_train, X_test, curr_y_train, y_test)
            score = aps_score(res['confusion_matrix'])
            res['score'] = score
            results[type(clf).__name__][var] = res
            measures_dict[i] = {technique: var, 'Classifier': type(clf).__name__, 'Price': score}
            i += 1


    measures = pd.DataFrame.from_dict(measures_dict, "index")
    measures.to_csv('plot_data/{}.csv'.format(filename))
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='Classifier', y='Price', hue=technique, data=measures)
    
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '{0:.3f}'.format(float(p.get_height())), 
            fontsize=12, color='black', ha='center', va='bottom')
    
    plt.savefig('images/{}.pdf'.format(filename))
    plt.clf()

    return results

def plot_param_improv(classifiers, X_data, y_data, y_test, params, param='Parameter', filename='result', style='darkgrid', figsize=(16,6)):
    sns.set(style=style)
    measures_dict = {}
    i = 0
    param_counter = 0
    results = {}
    for clf in classifiers:
        clf = clone(clf)
        param_value = params[param_counter]
        param_counter += 1
        results['{}={}'.format(param, param_value)] = {}
        for var in X_data:
            X_train, X_test = X_data[var]
            y_train = y_data[var]
            res = classifier_statistics(clf, X_train, X_test, y_train, y_test)
            score = aps_score(res['confusion_matrix'])
            res['score'] = score
            results['{}={}'.format(param, param_value)][var] = res
            measures_dict[i] = {param: param_value, 'Price': score, 'Transformation': var}
            i += 1

    measures = pd.DataFrame.from_dict(measures_dict, "index")
    measures.to_csv('plot_data/{}.csv'.format(filename))
    plt.figure(figsize=figsize)
    g = sns.FacetGrid(measures, hue="Transformation", size=8)
    g = g.map(plt.scatter, param, "Price").add_legend()
    g = g.map(plt.plot, param, "Price")
    g.axes[0,0].set_ylim(ymin=0)
    plt.savefig('images/{}.pdf'.format(filename))
    plt.clf()

    return results