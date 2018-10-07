import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

sn.set('talk', 'whitegrid', 'dark', font_scale=1, font='Ricty',
rc={"lines.linewidth": 2, 'grid.linestyle': '--'})


def sensibility(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tp / (tp + fn)

def specificity(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()

    return tn / (tn + fp)

bank = pd.read_csv('bank.csv')

# Binary encode gender to male
bank['male'] = bank['gender'].map({'MALE': 1, 'FEMALE': 0})
bank = bank.drop(columns='gender')

# One hot encoding region
region_dummies = pd.get_dummies(bank['region'], prefix='region')
bank = pd.concat([bank, region_dummies], axis=1)
bank = bank.drop(columns='region')

# Binary encode married
bank['married'] = bank['married'].map({'YES': 1, 'NO': 0})

# Binary encode car
bank['car'] = bank['car'].map({'YES': 1, 'NO': 0})

# Binary encode save_act
bank['save_act'] = bank['save_act'].map({'YES': 1, 'NO': 0})

# Binary encode current_act
bank['current_act'] = bank['current_act'].map({'YES': 1, 'NO': 0})

# Binary encode mortgage
bank['mortgage'] = bank['mortgage'].map({'YES': 1, 'NO': 0})

# Binart encode pep
bank['pep'] = bank['pep'].map({'YES': 1, 'NO': 0})

bank.head()

def exercise(classifier, classifier_name):
    print('Classifier {}'.format(classifier_name))
    X = bank.iloc[:, bank.columns != 'pep']
    y = bank['pep']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    
    
    classifier.fit(X_train, y_train)
    
    predicted = classifier.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, predicted)
    print('Accuracy {}'.format(accuracy_score(y_test, predicted)))
    #sn.heatmap(conf_matrix)
    print(conf_matrix)

    
    
    print('Sensibility: {}, specificity: {}'.format(sensibility(conf_matrix), specificity(conf_matrix)))
    
    fpr, tpr, _ = roc_curve(y_test.ravel(), predicted.ravel())
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve %s (area = %0.2f)' % (classifier_name, roc_auc))
    
    
exercise(GaussianNB(), 'Naive Bayes')
exercise(KNeighborsClassifier(n_neighbors=3), 'KNN')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()