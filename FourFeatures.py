
from scikit_weak.data_representation import ContinuousWeakLabel, FuzzyLabel, IntervalLabel
class GaussianFuzzyLabel(ContinuousWeakLabel, FuzzyLabel):
    def __init__(self, mean, std):
        if std < 0:
            raise ValueError("Std cannot be lower than 0")
        self.mean = mean
        self.std = std

    def sample_value(self):
        alpha = np.random.random()
        interval = self.get_cut(alpha)
        return interval.sample_value()

    def __eq__(self, other):
        if isinstance(other, GaussianFuzzyLabel):
            return (self.mean == other.mean) and (self.std == other.std)
        else:
            return False

    def __getitem__(self, val):
        return np.exp(-(self.mean - val)**2/(2*self.std**2))

    def get_cut(self, alpha):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha should be between 0 and 1")
        else:
            return IntervalLabel( self.mean - self.std*np.sqrt(-2*np.log(alpha)), self.mean + self.std*np.sqrt(-2*np.log(alpha)) )

    def __str__(self):
        return "Gaussian[%f, %f]" % (self.mean, self.std)

class RandomLabel(ContinuousWeakLabel):
    def __init__(self, distrib):
        self.distrib = distrib

    def sample_value(self):
        return self.distrib.rvs()

    def __eq__(self, other):
        if isinstance(other, RandomLabel):
            return (self.distrib == other.distrib)
        else:
            return False

    def __getitem__(self, val):
        return self.distrib.pdf(val)

    def __str__(self):
        return str(self.distrib)

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
class DataGenerator(BaseEstimator, TransformerMixin):

  def __init__(self):
    pass

  def fit(self, X):
    self.X = X
    return self

  def transform(self, X):
    X_res = np.empty(X.shape)
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        X_res[i,j] = X[i,j].sample_value()
    return X_res

  def fit_transform(self, X):
    self.fit(X)
    return self.transform(self.X)

#Classifiers

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class PertbClf1(BaseEstimator, ClassifierMixin):
  def __init__(self, estimator=None, random_state=None):
    self.estimator = estimator
    self.random_state = random_state

  def fit(self, X, y):
    self.X = X
    self.y = y
    state = np.random.get_state()
    if not (self.random_state is None):
      np.random.seed(self.random_state)

    self.augm_X = np.zeros((X.shape[0], X.shape[1]))
    self.augm_y = np.zeros(len(y))
    self.__sample_data()
    self.estimator.fit(self.augm_X, self.augm_y)
    if not (self.random_state is None):
      np.random.set_state(state)
    return self

  def __sample_data(self):
    for i in range(self.X.shape[0]):
      for j in range(self.X.shape[1]):
        self.augm_X[i,j] = self.X[i,j].sample_value()
        self.augm_y[i] = self.y[i]

  def __augment_predict_data(self, X):
    augm_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        augm_X[i,j] = X.iloc[i,j].sample_value()
    return augm_X

  def pred_compute(self, X, y_true):
    augm_X = self.__augment_predict_data(X)
    final_probas = self.estimator.predict_proba(augm_X)
    final_preds = self.estimator.predict(augm_X)
    acc = accuracy_score(y_true, final_preds)
    f1 = f1_score(y_true, final_preds)
    auc = roc_auc_score(y_true, final_probas[:, 1])
    return auc, acc, f1

class PertbClf2(BaseEstimator, ClassifierMixin):
  def __init__(self, estimator=None, random_state=None):
    self.estimator = estimator
    self.random_state = random_state

  def fit(self, X, y):
    self.X = X
    self.y = y
    state = np.random.get_state()
    if not (self.random_state is None):
      np.random.seed(self.random_state)

    self.augm_X = np.zeros((X.shape[0], X.shape[1]))
    self.augm_y = np.zeros(len(y))
    self.__sample_data()
    self.estimator.fit(self.augm_X, self.augm_y)
    if not (self.random_state is None):
      np.random.set_state(state)
    return self

  def __sample_data(self):
    for i in range(self.X.shape[0]):
      for j in range(self.X.shape[1]):
        self.augm_X[i,j] = self.X[i,j].sample_value()
        self.augm_y[i] = self.y[i]

  def pred_compute(self, X, y_true):
    final_probas = self.estimator.predict_proba(X)
    final_preds = self.estimator.predict(X)
    acc = accuracy_score(y_true, final_preds)
    f1 = f1_score(y_true, final_preds)
    auc = roc_auc_score(y_true, final_probas[:, 1])
    return auc, acc, f1

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

class AugClf(BaseEstimator, ClassifierMixin):
  def __init__(self, estimator=None, num_samples=100, random_state=None):
    self.estimator = estimator
    self.num_samples = num_samples
    self.random_state = random_state

  def fit(self, X, y):
    self.X = X
    self.y = y
    state = np.random.get_state()
    if not (self.random_state is None):
      np.random.seed(self.random_state)

    self.augm_X = np.zeros((X.shape[0]*self.num_samples, X.shape[1]))
    self.augm_y = np.zeros(len(y)*self.num_samples)
    self.__sample_data()
    self.estimator.fit(self.augm_X, self.augm_y)
    if not (self.random_state is None):
      np.random.set_state(state)
    return self

  def __sample_data(self):
    for i in range(self.X.shape[0]):
      for j in range(self.X.shape[1]):
        for t in range(self.num_samples):
          self.augm_X[i*self.num_samples + t, j] = self.X[i,j].sample_value()
          self.augm_y[i*self.num_samples + t] = self.y[i]

  def __augment_predict_data(self, X):
    augm_X = np.zeros((X.shape[0] * self.num_samples, X.shape[1]))
    for i in range(X.shape[0]):
      for j in range(X.shape[1]):
        for t in range(self.num_samples):
          try:
            augm_X[(i*self.num_samples) + t, j] = X.iloc[i,j].sample_value()
          except AttributeError:
            print(f"Error sampling value from X[{i}, {j}]")
            raise
    return augm_X

  def pred_compute(self, X, y_true):
    augm_X = self.__augment_predict_data(X)
    augmented_probas = self.estimator.predict_proba(augm_X)
    final_probas = np.zeros((X.shape[0], augmented_probas.shape[1]))
    for i in range(X.shape[0]):
        final_probas[i] = np.mean(augmented_probas[i * self.num_samples: (i + 1) * self.num_samples], axis=0)

    final_preds = np.argmax(final_probas, axis=1)
    acc = accuracy_score(y_true, final_preds)
    f1 = f1_score(y_true, final_preds)
    auc = roc_auc_score(y_true, final_probas[:, 1])
    return auc, acc, f1

#ML-Prep

features = ['Leukocytes', 'Platelets', 'Eosinophils', 'Monocytes']

import pandas as pd
X_test = pd.read_csv('Dataset-1c.csv')
y_test = X_test['CoV-2'].values.astype(np.int8)
X_test = X_test[features].copy()

X_train = pd.read_csv('Dataset-2b.csv')
y_train = X_train['CoV-2'].values.astype(np.int8)
X_train = X_train[features].copy()
spw = (sum(y_train==0))/(sum(y_train==1))
eval_set = [(X_train, y_train), (X_test, y_test)]

variabilities = [[0.019, 0.111, 0.033],
                 [0.01015, 0.0325, 0.0722],
                 [0.0925, 0.141, 0.156],
                 [0.046, 0.104, 0.134]]
columns = ["CVA", "CVI(s)", "CVI(m)"]
vars = pd.DataFrame(variabilities, columns=columns, index=features)

import random
from scipy import stats
from xgboost import XGBClassifier
from sklearn.impute import KNNImputer

epsilon = 0.001
times = 100

#Perturbated - All Xs

res_sampl = pd.DataFrame(columns=['AUC Score', 'F1 Score', 'Accuracy'])
res_sampl_base = pd.DataFrame(columns=['AUC Score', 'F1 Score', 'Accuracy'])

for t in range(times):
  print(f"Iteration {t+1}")
  imp = KNNImputer()
  X_train_temp = X_train.copy()
  X_test_temp = X_test.copy()
  X_train_temp.loc[:,:] = imp.fit_transform(X_train_temp.values)
  X_test_temp.loc[:,:] = imp.fit_transform(X_test_temp.values)
  X_train_ran = X_train_temp.copy()
  X_test_ran = X_test_temp.copy()
  dg = DataGenerator()
  X_train_sam = X_train_temp.copy()
  X_test_sam = X_test_temp.copy()
  print("Set Test Sam - Prepare Data for Sampling")
  for r in range(X_test_temp.shape[0]):
    for c in range(X_test_temp.shape[1]):
      std = (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      X_test_sam.iloc[r,c] = RandomLabel(stats.norm(loc=X_test_temp.iloc[r,c], scale = std ))

  X_train_sampl = X_train_temp.copy()
  X_test_sampl_base = X_test_temp.copy()
  X_test_sampl = X_test_temp.copy()
  X_test_sampl.loc[:,:] = dg.fit_transform(X_test_sam.values)
  X_train_fuzzy = X_train_temp.copy()
  X_train_fuzzy.loc[:,:] = np.empty(X_train_temp.shape, dtype=GaussianFuzzyLabel)
  X_test_fuzzy = X_test_temp.copy()
  X_test_fuzzy.loc[:,:] = np.empty(X_test_temp.shape, dtype=GaussianFuzzyLabel)
  X_test_fuzzy_base = X_test_temp.copy()
  X_test_fuzzy_base.loc[:,:] = np.empty(X_test_temp.shape, dtype=GaussianFuzzyLabel)
  for r in range(X_train_sampl.shape[0]):
    for c in range(X_test_sampl.shape[1]):
      std = (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVA"])**2
      if y_train[r] == 0:
        std += (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVI(s)"])**2
      else:
        std += (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_train_fuzzy.iloc[r,c] = GaussianFuzzyLabel(X_train_sampl.iloc[r,c], std)

  for r in range(X_test_sampl.shape[0]):
    for c in range(X_test_sampl.shape[1]):
      std = (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_test_fuzzy.iloc[r,c] = GaussianFuzzyLabel(X_test_sampl.iloc[r,c], std)

      std = (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_test_fuzzy_base.iloc[r,c] = GaussianFuzzyLabel(X_test_sampl_base.iloc[r,c], std)

  clf = PertbClf1(XGBClassifier(scale_pos_weight=spw, eval_metric=["error", "logloss"], eval_set=eval_set), random_state=99)
  clf.fit(X_train_fuzzy.values, y_train)
  AUC, ACC, F1 = clf.pred_compute(X_test_fuzzy, y_test)
  res_sampl.loc[len(res_sampl.index)]=[AUC, ACC, F1]
  AUC, ACC, F1 = clf.pred_compute(X_test_fuzzy_base, y_test)
  res_sampl_base.loc[len(res_sampl_base.index)]=[AUC, ACC, F1]
  print()

Metric_Means = []
Metric_StDevs = []
for col in (res_sampl.columns):
  Metric_Means.append(res_sampl[col].mean())
  Metric_StDevs.append(res_sampl[col].std())

Res = pd.DataFrame(index=(res_sampl.columns.values))
Res['Augmented Mean'] = Metric_Means
Res['Augmented StDev'] = Metric_StDevs

Metric_Means = []
Metric_StDevs = []
for col in (res_sampl_base.columns):
  Metric_Means.append(res_sampl_base[col].mean())
  Metric_StDevs.append(res_sampl_base[col].std())

Res['Augmented Mean (Baseline)'] = Metric_Means
Res['Augmented StDev (Baseline)'] = Metric_StDevs
Res

Res.to_csv("Four-Prtrb1.csv")

#Perturbated - Only X_train

res_sampl = pd.DataFrame(columns=['AUC Score', 'F1 Score', 'Accuracy'])
res_sampl_base = pd.DataFrame(columns=['AUC Score', 'F1 Score', 'Accuracy'])

for t in range(times):
  print(f"Iteration {t+1}")
  imp = KNNImputer()
  X_train_temp = X_train.copy()
  X_test_temp = X_test.copy()
  X_train_temp.loc[:,:] = imp.fit_transform(X_train_temp.values)
  X_test_temp.loc[:,:] = imp.fit_transform(X_test_temp.values)
  X_train_ran = X_train_temp.copy()
  X_test_ran = X_test_temp.copy()
  dg = DataGenerator()
  X_train_sam = X_train_temp.copy()
  X_test_sam = X_test_temp.copy()
  print("Set Test Sam - Prepare Data for Sampling")
  for r in range(X_test_temp.shape[0]):
    for c in range(X_test_temp.shape[1]):
      std = (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      X_test_sam.iloc[r,c] = RandomLabel(stats.norm(loc=X_test_temp.iloc[r,c], scale = std ))

  X_train_sampl = X_train_temp.copy()
  X_test_sampl_base = X_test_temp.copy()
  X_test_sampl = X_test_temp.copy()
  X_test_sampl.loc[:,:] = dg.fit_transform(X_test_sam.values)
  X_train_fuzzy = X_train_temp.copy()
  X_train_fuzzy.loc[:,:] = np.empty(X_train_temp.shape, dtype=GaussianFuzzyLabel)
  X_test_fuzzy = X_test_temp.copy()
  X_test_fuzzy.loc[:,:] = np.empty(X_test_temp.shape, dtype=GaussianFuzzyLabel)
  X_test_fuzzy_base = X_test_temp.copy()
  X_test_fuzzy_base.loc[:,:] = np.empty(X_test_temp.shape, dtype=GaussianFuzzyLabel)
  for r in range(X_train_sampl.shape[0]):
    for c in range(X_test_sampl.shape[1]):
      std = (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVA"])**2
      if y_train[r] == 0:
        std += (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVI(s)"])**2
      else:
        std += (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_train_fuzzy.iloc[r,c] = GaussianFuzzyLabel(X_train_sampl.iloc[r,c], std)

  for r in range(X_test_sampl.shape[0]):
    for c in range(X_test_sampl.shape[1]):
      std = (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_test_fuzzy.iloc[r,c] = GaussianFuzzyLabel(X_test_sampl.iloc[r,c], std)

      std = (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_test_fuzzy_base.iloc[r,c] = GaussianFuzzyLabel(X_test_sampl_base.iloc[r,c], std)

  clf = PertbClf2(XGBClassifier(scale_pos_weight=spw, eval_metric=["error", "logloss"], eval_set=eval_set), random_state=99)
  clf.fit(X_train_fuzzy.values, y_train)
  AUC, ACC, F1 = clf.pred_compute((X_test_sampl.values), y_test)
  res_sampl.loc[len(res_sampl.index)]=[AUC, ACC, F1]
  AUC, ACC, F1 = clf.pred_compute((X_test_sampl_base.values), y_test)
  res_sampl_base.loc[len(res_sampl_base.index)]=[AUC, ACC, F1]
  print()

res_sampl
res_sampl.to_csv("4allI-Prtrb.csv")

res_sampl_base
res_sampl_base.to_csv("4allI-Prtrb_base.csv")

Metric_Means = []
Metric_StDevs = []
for col in (res_sampl.columns):
  Metric_Means.append(res_sampl[col].mean())
  Metric_StDevs.append(res_sampl[col].std())

Res = pd.DataFrame(index=(res_sampl.columns.values))
Res['Augmented Mean'] = Metric_Means
Res['Augmented StDev'] = Metric_StDevs

Metric_Means = []
Metric_StDevs = []
for col in (res_sampl_base.columns):
  Metric_Means.append(res_sampl_base[col].mean())
  Metric_StDevs.append(res_sampl_base[col].std())

Res['Augmented Mean (Baseline)'] = Metric_Means
Res['Augmented StDev (Baseline)'] = Metric_StDevs
Res

Res.to_csv("Four-Prtrb2.csv")

#Augmented

res_sampl = pd.DataFrame(columns=['AUC Score', 'F1 Score', 'Accuracy'])
res_sampl_base = pd.DataFrame(columns=['AUC Score', 'F1 Score', 'Accuracy'])

for t in range(times):
  print(f"Iteration {t+1}")
  imp = KNNImputer()
  X_train_temp = X_train.copy()
  X_test_temp = X_test.copy()
  X_train_temp.loc[:,:] = imp.fit_transform(X_train_temp.values)
  X_test_temp.loc[:,:] = imp.fit_transform(X_test_temp.values)
  X_train_ran = X_train_temp.copy()
  X_test_ran = X_test_temp.copy()
  dg = DataGenerator()
  X_train_sam = X_train_temp.copy()
  X_test_sam = X_test_temp.copy()
  print("Set Test Sam - Prepare Data for Sampling")
  for r in range(X_test_temp.shape[0]):
    for c in range(X_test_temp.shape[1]):
      std = (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_temp.iloc[r,c]*vars.loc[X_test_temp.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      X_test_sam.iloc[r,c] = RandomLabel(stats.norm(loc=X_test_temp.iloc[r,c], scale = std ))

  X_train_sampl = X_train_temp.copy()
  X_test_sampl_base = X_test_temp.copy()
  X_test_sampl = X_test_temp.copy()
  X_test_sampl.loc[:,:] = dg.fit_transform(X_test_sam.values)
  X_train_fuzzy = X_train_temp.copy()
  X_train_fuzzy.loc[:,:] = np.empty(X_train_temp.shape, dtype=GaussianFuzzyLabel)
  X_test_fuzzy = X_test_temp.copy()
  X_test_fuzzy.loc[:,:] = np.empty(X_test_temp.shape, dtype=GaussianFuzzyLabel)
  X_test_fuzzy_base = X_test_temp.copy()
  X_test_fuzzy_base.loc[:,:] = np.empty(X_test_temp.shape, dtype=GaussianFuzzyLabel)
  X_train_ran = X_train_temp.copy()
  X_train_ran.loc[:,:] = np.empty(X_train_temp.shape, dtype=RandomLabel)
  X_test_ran = X_test_temp.copy()
  X_test_ran.loc[:,:] = np.empty(X_test_temp.shape, dtype=RandomLabel)
  X_test_ran_base = X_test_temp.copy()
  X_test_ran_base.loc[:,:] = np.empty(X_test_temp.shape, dtype=RandomLabel)
  for r in range(X_train_sampl.shape[0]):
    for c in range(X_test_sampl.shape[1]):
      std = (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVA"])**2
      if y_train[r] == 0:
        std += (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVI(s)"])**2
      else:
        std += (X_train_sampl.iloc[r,c]*vars.loc[X_train_sampl.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_train_fuzzy.iloc[r,c] = GaussianFuzzyLabel(X_train_sampl.iloc[r,c], std)
      X_train_ran.iloc[r,c] = RandomLabel(stats.norm(loc=X_train_sampl.iloc[r,c], scale = std))

  for r in range(X_test_sampl.shape[0]):
    for c in range(X_test_sampl.shape[1]):
      std = (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_sampl.iloc[r,c]*vars.loc[X_test_sampl.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_test_fuzzy.iloc[r,c] = GaussianFuzzyLabel(X_test_sampl.iloc[r,c], std)
      X_test_ran.iloc[r,c] = RandomLabel(stats.norm(loc=X_test_sampl.iloc[r,c], scale = std))
      std = (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVA"])**2
      if y_test[r] == 0:
        std += (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVI(s)"])**2
      else:
        std += (X_test_sampl_base.iloc[r,c]*vars.loc[X_test_sampl_base.columns[c],"CVI(m)"])**2
      std = np.sqrt(std)
      if std < epsilon:
        std = epsilon
      X_test_fuzzy_base.iloc[r,c] = GaussianFuzzyLabel(X_test_sampl_base.iloc[r,c], std)
      X_test_ran_base.iloc[r,c] = RandomLabel(stats.norm(loc=X_test_sampl_base.iloc[r,c], scale = std))

  clf = AugClf(XGBClassifier(scale_pos_weight=spw, eval_metric=["error", "logloss"], eval_set=eval_set), random_state=99)
  clf.fit(X_train_fuzzy.values, y_train)
  AUC, ACC, F1 = clf.pred_compute(X_train_fuzzy, y_train)
  res_sampl.loc[len(res_sampl.index)]=[AUC, ACC, F1]
  AUC, ACC, F1 = clf.pred_compute(X_test_fuzzy_base, y_test)
  res_sampl_base.loc[len(res_sampl_base.index)]=[AUC, ACC, F1]
  print()

res_sampl
res_sampl.to_csv("4allI-Augm.csv")

res_sampl_base
res_sampl_base.to_csv("4allI-Augm_base.csv")

Metric_Means = []
Metric_StDevs = []
for col in (res_sampl.columns):
  Metric_Means.append(res_sampl[col].mean())
  Metric_StDevs.append(res_sampl[col].std())

Res = pd.DataFrame(index=(res_sampl.columns.values))
Res['Augmented Mean'] = Metric_Means
Res['Augmented StDev'] = Metric_StDevs

Metric_Means = []
Metric_StDevs = []
for col in (res_sampl_base.columns):
  Metric_Means.append(res_sampl_base[col].mean())
  Metric_StDevs.append(res_sampl_base[col].std())

Res['Augmented Mean (Baseline)'] = Metric_Means
Res['Augmented StDev (Baseline)'] = Metric_StDevs
Res

Res.to_csv("Four-Augmt.csv")

