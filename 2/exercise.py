import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
import seaborn as sns

def knn_classifier(n_neighbors):
    print('KNN with {} neighbor'.format(n_neighbors))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    return knn, knn.predict(X_test)

def accuracy(predicted, expected):
    samples = len(predicted)
    correct = 0
    for i in range(samples):
        if predicted[i] == expected[i]:
            correct += 1
    
    return (correct / samples) * 100

def true_false_negatives(cl, predicted, expected):
    samples = len(predicted)
    
    fp = 0
    fn = 0
    for i in range(samples):
        if predicted[i] == cl:
            if predicted[i] != expected[i]:
                fp += 1
        elif expected[i] == cl:
            if predicted[i] != expected[i]:
                fn += 1

    return fp, fn


iris = pd.read_csv('iris.csv')

X = iris.iloc[:,:4].values
y = iris[['class']].values.reshape(-1,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 1

print("KNN classifier")

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

predicted = knn.predict(X_test)

# a)

print('Accuracy: {}'.format(accuracy(predicted, y_test)))

# b)

fp, fn = true_false_negatives('Iris-virginica', predicted, y_test)

print('False positives: {}, False negatives: {}'.format(fp, fn))

# c)

fp, fn = true_false_negatives('Iris-setosa', predicted, y_test)

print('False positives: {}, False negatives: {}'.format(fp, fn))

# d)

scores = cross_val_score(knn, X, y, cv=10)

print('Cross validation scores: {}'.format(scores))

print()
print()


# 2

print("Naive Bayes classifier")

nbayes = GaussianNB()
nbayes.fit(X_train, y_train)

predicted = nbayes.predict(X_test)

# a)

print('Accuracy: {}'.format(accuracy(predicted, y_test)))

# b)

fp, fn = true_false_negatives('Iris-virginica', predicted, y_test)

print('False positives: {}, False negatives: {}'.format(fp, fn))

# c)

fp, fn = true_false_negatives('Iris-setosa', predicted, y_test)

print('False positives: {}, False negatives: {}'.format(fp, fn))

# d)

scores = cross_val_score(nbayes, X, y, cv=10)

print('Cross validation scores: {}'.format(scores))

print()
print()


# 3


glass = pd.read_csv('glass.csv')

X = glass.iloc[:,:4].values
y = glass[['Type']].values.reshape(-1,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


knn, predicted = knn_classifier(1)

# a)

print('Accuracy: {}'.format(accuracy(predicted, y_test)))

# b)
neighbors = [5, 10, 15, 50, 100]
accuracies = []
for n in neighbors:
    knn, predicted = knn_classifier(n)
    accur = accuracy(predicted, y_test)
    accuracies.append(accur)
    print('Accuracy: {}'.format(accur))

results = pd.DataFrame({
            'accuracy': accuracies,
            'neighbors': neighbors
        })


# c)
    
sns.lmplot('neighbors', 'accuracy', data=results, fit_reg=False)

# d)

for n in neighbors:
    knn, _ = knn_classifier(n)
    scores = cross_val_score(knn, X, y, cv=10)
    print('Cross validation scores: {}'.format(scores))
    
# e)

