import pandas as pd

breast_cancer = pd.read_csv('breast_cancer.csv')

breast_cancer['irradiat'] = breast_cancer['irradiat'].map({"'yes'": 1, "'no'": 0})