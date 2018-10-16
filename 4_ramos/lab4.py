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


def ex1():

    bc = pd.read_csv('breast_cancer.csv')

    le = preprocessing.LabelEncoder()
    for i in range(10):
        bc.iloc[:,i] = le.fit_transform(bc.iloc[:,i])

    clf = tree.DecisionTreeClassifier()
    #X = le[:, bc.columns != 'Class']
    #y = le['Class']
    print(list(le.classes_))
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #clf.fit(X_train, y_train)

    #predicted = clf.predict(X_test)

    #conf_matrix = confusion_matrix(y_test, predicted)
    #print('Accuracy {}'.format(accuracy_score(y_test, predicted)))

    #print(conf_matrix)

ex1()