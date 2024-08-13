#!pip uninstall xgboost -y
#!pip install xgboost==0.90

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score, fbeta_score, brier_score_loss, recall_score

def computemetrics(y_true, y_proba):
  prop = len(y_true[y_true == 1])/len(y_true)
  y_pred = (y_proba >= (0.5)).astype(int)
  cm = confusion_matrix(y_true, y_pred)
  print((dfn[i]), ';;;',
        round((((recall_score(y_true, y_pred))*prop - (1-(recall_score(y_true, y_pred, pos_label=0)))*(1-prop)*(0.5)/(1-(0.5)))/prop), 3),
        ';;;', round(roc_auc_score(y_true, tsprbs), 3), ';;;',
        round(balanced_accuracy_score(y_true, y_pred), 3), ';;;',
        round(f1_score(y_true, y_pred), 3), ';;;',
        round(fbeta_score(y_true, y_pred, beta=2), 3), ';;;',
        round(brier_score_loss(y_true, y_pred), 3), ';;;',
        round(((cm[0,0]+cm[1,1])/(sum(sum(cm)))), 3), ';;;',
        round((cm[1,1]/(cm[1,0]+cm[1,1])), 3), ';;;',
        round((cm[0,0]/(cm[0,0]+cm[0,1])), 3))

dfn = ['1','1a','1b','1c','2a','2b','3','3a','3b']
print('Dataset ;;; Standardized Net Benefit ;;; AUC Score ;;; Balanced Accuracy ;;; F1 Score ;;; F2 Score ;;; Brier Score ;;; Accuracy ;;; Sensitivity ;;; Specificity')
for i in (range(0,9)):
  d = pd.read_csv('datasets/Dataset-{}.csv'.format(dfn[i]))
  y = d[['CoV-2']].copy()
  X = d.copy()
  X.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)
  spw = (sum(d['CoV-2']==0))/(sum(d['CoV-2']==1))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
  model = XGBClassifier(scale_pos_weight=spw)
  eval_set = [(X_train, y_train.values.ravel()), (X_test, y_test.values.ravel())]
  model.fit(X_train, y_train.values.ravel(), eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
  model.save_model('XGB/Model-{}.json'.format(dfn[i]))
  tsprbs = np.array((model.predict_proba(X_test))[:, 1])
  computemetrics((y_test.to_numpy()), tsprbs)
