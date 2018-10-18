import pandas as pd
import numpy as np


def transform_dataset(dataset):
    table = dataset
    for col in table:
        if table[col].dtype == np.object:
            col_dummies = pd.get_dummies(table[col], prefix=col, drop_first=True)
            table = pd.concat([table, col_dummies], axis=1)
            table = table.drop(columns=col)
    return table
