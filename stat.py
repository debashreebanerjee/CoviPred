import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan)

lst = ['Dataset-1a', 'Dataset-2a', 'Dataset-3']

for l in lst:
  d = pd.read_csv('datasets/{}.csv'.format(l))
  y = d[['CoV-2']].copy()
  d.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)
  imp.fit(d)
  x = pd.DataFrame(imp.transform(d))
  x.columns = d.columns
  x.index = d.index
  cols = x.columns.tolist()
  for c in cols:
    print(c, ";;;", pointbiserialr(x[c], y['CoV-2']))
  print('\n\n')



from scipy.stats import kstest
def listKS(df1, df2):
  ksstats = []
  cols = (df1.columns)
  for col in cols:
    print(kstest((df1[col]), (df2[col])))

df14 = pd.read_csv('datasets/Dataset-1c.csv')
df24 = pd.read_csv('datasets/Dataset-2b.csv')
listKS(df14, df24)

df114 = pd.read_csv('datasets/Dataset-1b.csv')
df314 = pd.read_csv('datasets/Dataset-3b.csv')
listKS(df114, df314)
