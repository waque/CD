import pandas as pd
import numpy as np
import arff
import sys
import seaborn as sns
sys.path.append('..')
from utils_cd import (
    transform_X
)

zoo_data = open('zoo.arff')
zoo_data = arff.load(zoo_data)

print(zoo_data)
