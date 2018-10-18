import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn import tree
import graphviz
from sklearn.ensemble import RandomForestClassifier
from utils import transform_dataset

cancer = pd.read_csv('breast_cancer.csv')

cancer_transformed = transform_dataset(cancer)