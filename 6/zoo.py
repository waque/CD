import pandas as pd
import numpy as np
import arff
import sys
import re
sys.path.append('..')
from utils_cd import (
        transform_X
)
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
from pymining import seqmining
from prefixspan import PrefixSpan

data = arff.load(open('zoo.arff'))
attrs = []

for attr in data['attributes']:
        attrs.append(attr[0])

df = pd.DataFrame(data=data['data'], columns=attrs)
df.fillna('-1', inplace=True)

for col in list(df):
        attrs = []
        values = df[col].unique().tolist()
        values.sort()
        
        for value in values:
                attrs.append('{}:{}'.format(col, value))
        lb = LabelBinarizer().fit_transform(df[col])

        if len(attrs) == 2:
                v = list(map(lambda x: 1 - x, lb))
                lb = np.concatenate((lb,v), 1)

        df2 = pd.DataFrame(data=lb, columns=attrs)
        if '-1' in values:
                df2 = df2.drop(columns=["{}:-1".format(col)])

        df = df.drop(columns=[col])
        df = pd.concat([df, df2], axis=1, join='inner')

treshholds = np.arange(0.4, 0.9, 0.1)

for sup in supports:
        print('Testing {} treshold'.format(sup))
        frequent_itemsets = apriori(df, min_support=0.5)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
        print(rules)


fp = open('fifa.txt')
line = fp.readline()
seqdata = []
while line:
        seqdata.append(line.strip().split(' '))
        line = fp.readline()
fp.close()
freq_seqs = seqmining.freq_seq_enum(seqdata, 550)
sorted(freq_seqs)
ps = PrefixSpan(seqdata)
print(ps.frequent(500, closed=True), '\n')
print(ps.topk(5, closed=True))