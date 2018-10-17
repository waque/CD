import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


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
        
    return table.head()

def ex1():
    table = pd.read_csv('breast_cancer.csv')
    table_tans = pd.get_dummies(table, ['age','menopause','tumor_size','inv_nodes','node_caps','deg_malig','breast','breast_quad','irradiat','Class'], drop_first=True)
    
    X = table_tans.iloc[:, table.columns != "Class_'recurrence-events'"]
    
    print(y)
    #clf = tree.DecisionTreeClassifier()
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #clf.fit(X_train, y_train)

    #predicted = clf.predict(X_test)

    #conf_matrix = confusion_matrix(y_test, predicted)
    #print('Accuracy {}'.format(accuracy_score(y_test, predicted)))

    #print(conf_matrix)
    

ex1()

