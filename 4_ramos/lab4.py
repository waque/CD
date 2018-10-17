import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn import tree
import graphviz
from sklearn.ensemble import RandomForestClassifier


def sensibility(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tp / (tp + fn)

def specificity(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tn / (tn + fp)

def transform_table(file, columns):
    table = pd.read_csv(file)
    for col in columns:
        col_dummies = pd.get_dummies(table[col], prefix=col)
        table = pd.concat([table, col_dummies], axis=1)
        table = table.drop(columns=col)
        
    return table

cancer = pd.read_csv('breast_cancer.csv')

# encoding age
age_dummies = pd.get_dummies(cancer['age'], prefix='age')
cancer = pd.concat([cancer, age_dummies], axis=1)
cancer = cancer.drop(columns='age')

# encoding menopause
menopause_dummies = pd.get_dummies(cancer['menopause'], prefix='menopause')
cancer = pd.concat([cancer, menopause_dummies], axis=1)
cancer = cancer.drop(columns='menopause')

# encoding tumor_size
tumor_size_dummies = pd.get_dummies(cancer['tumor_size'], prefix='tumor_size')
cancer = pd.concat([cancer, tumor_size_dummies], axis=1)
cancer = cancer.drop(columns='tumor_size')

# encoding inv_nodes
inv_nodes_dummies = pd.get_dummies(cancer['inv_nodes'], prefix='inv_nodes')
cancer = pd.concat([cancer, inv_nodes_dummies], axis=1)
cancer = cancer.drop(columns='inv_nodes')

# encoding node_caps
cancer['node_caps'] = pd.get_dummies(cancer['node_caps'], drop_first=True)

#encoding deg_malig
deg_malig_dummies = pd.get_dummies(cancer['deg_malig'], prefix='dg')
cancer = pd.concat([cancer, deg_malig_dummies], axis=1)
cancer = cancer.drop(columns='deg_malig')

# encoding right_breast
cancer['breast'] = pd.get_dummies(cancer['breast'], drop_first=True)

#encoding breast_quad
breast_quad_dummies = pd.get_dummies(cancer['breast_quad'], prefix='bq')
cancer = pd.concat([cancer, breast_quad_dummies], axis=1)
cancer = cancer.drop(columns='breast_quad')

# encoding node_caps
cancer['irradiat'] = pd.get_dummies(cancer['irradiat'], drop_first=True)

# encoding node_caps
cancer['Class'] = pd.get_dummies(cancer['Class'], drop_first=True)

cancer.head()

def ex1():
    X = cancer.iloc[:, cancer.columns != 'Class']
    y = cancer['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #best splitter 
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predicted)
    print('Accuracy best splitter {}'.format(accuracy_score(y_test, predicted)))

    #random splitter
    clf = tree.DecisionTreeClassifier(splitter='random')
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predicted)
    print('Accuracy random splitter {}'.format(accuracy_score(y_test, predicted)))
    dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True,special_characters=True) 
    graph = graphviz.Source(dot_data) 
    graph.render("cancer") 

def ex2_aux(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predicted)
    return predicted
    
def ex2():

    X = cancer.iloc[:, cancer.columns != 'Class']
    y = cancer['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #default
    clf = RandomForestClassifier()
    sum = 0
    for i in range(0,100):
        sum += accuracy_score(y_test, ex2_aux(clf, X_train, X_test, y_train, y_test))    
    print('Average Accuracy random forest default' + str(sum/100))

    #default
    clf = RandomForestClassifier(n_estimators=15)
    sum = 0
    for i in range(0,100):
        sum += accuracy_score(y_test, ex2_aux(clf, X_train, X_test, y_train, y_test))
    print('Average Accuracy random forest 50 trees' + str(sum/100))


def ex3():
    credit = pd.read_csv('credit.csv')

    #REKT