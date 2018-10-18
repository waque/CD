import pandas as pd
import sys
sys.path.append('..')
from utils_cd import split_dataset_transformed

breast_cancer = pd.read_csv('breast_cancer.csv')

X, y = split_dataset_transformed(breast_cancer, 'Class')
