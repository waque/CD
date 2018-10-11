import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE




def sensibility(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()

    return tp / (tp + fn)

def specificity(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()

    return tn / (tn + fp)


unbalanced = pd.read_csv('unbalanced.csv')

unbalanced['Outcome'] = unbalanced['Outcome'].map({'Active': 1, 'Inactive': 0})

X = unbalanced.iloc[:, unbalanced.columns != 'Outcome']
y = unbalanced['Outcome']
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def exercise(classifier, classifier_name):
    classifier.fit(X_train, y_train)
    
    predicted = classifier.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, predicted)
    print(conf_matrix)
    print('Accuracy {}'.format(accuracy_score(y_test, predicted)))
    
    print('Sensibility: {}, specificity: {}'.format(sensibility(conf_matrix), specificity(conf_matrix)))
    
    fpr, tpr, _ = roc_curve(y_test.ravel(), predicted.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve %s (area = %0.2f)' % (classifier_name, roc_auc))
    
    
    
exercise(GaussianNB(), 'Naive Bayes')
exercise(KNeighborsClassifier(n_neighbors=3), 'KNN')

for i in [1, 10, 100]:
    exercise(KNeighborsClassifier(n_neighbors=i), 'KNN {}'.format(i))


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# Balancing data

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, recall_score, classification_report

parameters = {
    'n_neighbors': [1, 10, 100]
             }
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters, cv=5, verbose=5, n_jobs=3)
clf.fit(X_train_res, y_train_res.ravel())

clf.best_params_

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train_res, y_train_res.ravel())



y_train_pre = knn1.predict(X_train)
print(confusion_matrix(y_train, y_train_pre))


