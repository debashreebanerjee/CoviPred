def computemetrics(y_true, y_proba):
  thresh = [0.15, 0.5]
  prop = len(y_true[y_true == 1])/len(y_true)
  print('Threshold ;;; Standardized Net Benefit ;;; AUC Score ;;; Balanced Accuracy ;;; F1 Score ;;; F2 Score ;;; Brier Score ;;; Accuracy ;;; Sensitivity ;;; Specificity')
  for th in thresh:
    y_pred = (y_proba >= th).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(th, ';;;',
          round((((recall_score(y_true, y_pred))*prop - (1-(recall_score(y_true, y_pred, pos_label=0)))*(1-prop)*th/(1-th))/prop), 3),
          ';;;', round(roc_auc_score(y_true, tsprbs), 3), ';;;',
          round(balanced_accuracy_score(y_true, y_pred), 3), ';;;',
          round(f1_score(y_true, y_pred), 3), ';;;',
          round(fbeta_score(y_true, y_pred, beta=2), 3), ';;;',
          round(brier_score_loss(y_true, y_pred), 3), ';;;',
          round(((cm[0,0]+cm[1,1])/(sum(sum(cm)))), 3), ';;;',
          round((cm[1,1]/(cm[1,0]+cm[1,1])), 3), ';;;',
          round((cm[0,0]/(cm[0,0]+cm[0,1])), 3))



X_test = pd.read_csv('datasets/Dataset-1c.csv')
y_test = X_test['CoV-2'].to_numpy()
X_test.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)

X_train = pd.read_csv('datasets/Dataset-2b.csv')
y_train = X_train[['CoV-2']].copy()
X_train.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)

eval_set = [(X_train, y_train.values.ravel()), (X_test, y_test.ravel())]
finalmodel = XGBClassifier(eval_metric=["error", "logloss"], eval_set=eval_set)
finalmodel.fit(X_train, y_train.values.ravel(), verbose=False)
finalmodel.save_model('ExtVal/2-4Feat(XGB-0.90).json')

tsprbs = np.array((finalmodel.predict_proba(X_test))[:, 1])
computemetrics(y_test, tsprbs)

out = pd.DataFrame()
out['y_true'] = y_test
out['y_proba'] = tsprbs
out.to_csv('ExtVal/2-4F × 1-4F.csv', index=False)



X_train = pd.read_csv('datasets/Dataset-3b.csv')
y_train = X_train[['CoV-2']].copy()
X_train.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)

X_test = pd.read_csv('datasets/Dataset-1b.csv')
y_test = X_test['CoV-2'].to_numpy()
X_test.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)

eval_set = [(X_train, y_train.values.ravel()), (X_test, y_test.ravel())]
finalmodel = XGBClassifier(eval_metric=["error", "logloss"], eval_set=eval_set)
finalmodel.fit(X_train, y_train.values.ravel(), verbose=False)
finalmodel.save_model('ExtVal/3-14Feat(XGB-0.90).json')

tsprbs = np.array((finalmodel.predict_proba(X_test))[:, 1])
computemetrics(y_test, tsprbs)

out = pd.DataFrame()
out['y_true'] = y_test
out['y_proba'] = tsprbs
out.to_csv('ExtVal/3-14F × 1-14F.csv', index=False)



model = XGBClassifier()
model.load_model('XGB/Model-2b.json')

dn = ['Advanced','Early']
print('Dataset ;;; Accuracy ;;; Sensitivity ;;; Misclassification ;;; True Positive ;;; False Positive ;;; True Negative ;;; False Negative')
for i in range(0,2):
  d = pd.read_csv('rawdata/{}.csv'.format(dn[i]))
  out = pd.DataFrame()
  out['y_true'] = (np.ones(((len(d)),), dtype=int))
  out['y_proba'] = (model.predict_proba(d))[:, 1]
  out.to_csv('ExtVal/2-4F × {}.csv'.format(dn[i]), index=False)
  pred = ((out['y_proba']) >= (0.5)).astype(int)
  cm = confusion_matrix(out['y_true'], pred)
  print((dn[i]), ';;;', ((cm[0,0]+cm[1,1])/(sum(sum(cm)))), ';;;', (cm[1,1]/(cm[1,0]+cm[1,1])), ';;;', ((cm[0,1]+cm[1,0])/(sum(sum(cm)))), ';;;', (cm[0,0]), ';;;', (cm[1,0]), ';;;', (cm[1,1]), ';;;', (cm[0,1]))
