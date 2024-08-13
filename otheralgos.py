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

dfn = ['1','1a','1b','1c','2a','2b','3','3a','3b']
results=pd.DataFrame(columns=['Dataset','Model','Accuracy','Sensitivity','Specificity','AUC Score','True Positive','False Positive','True Negative','False Negative'])
imp = IterativeImputer(missing_values=np.nan)

for i in (range(0,9)):
  d = pd.read_csv('datasets/Dataset-{}.csv'.format(dfn[i]))
  y = d[['CoV-2']].copy()
  x = d.copy()
  x.drop(labels=None, axis=1, columns=['CoV-2'], level=None, inplace=True)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
  random_probs = [0 for b in range(len(y_test))]
  p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
  #
  spw = (sum(d['CoV-2']==0))/(sum(d['CoV-2']==1))
  model = XGBClassifier(scale_pos_weight=spw)
  eval_set = [(x_train, y_train.values.ravel()), (x_test, y_test.values.ravel())]
  model.fit(x_train, y_train.values.ravel(), eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)
  pred_prob = model.predict_proba(x_test)
  fpr, tpr, thresh = roc_curve(y_test, (pred_prob[:, 1]), pos_label=1)
  #
  imp.fit(x)
  X_train = imp.transform(x_train)
  X_test = imp.transform(x_test)
  #
  modelLR = LogisticRegression(random_state=0, max_iter=1000)
  modelLR.fit(X_train, y_train.values.ravel())
  pred_prob1 = modelLR.predict_proba(X_test)
  fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
  y_pred=modelLR.predict(X_test)
  cm=confusion_matrix(y_test,y_pred)
  auc=(roc_auc_score(y_test, (pred_prob1[:, 1])))
  results.loc[len(results.index)]=[dfn[i],'Logistic Regression',((cm[0,0]+cm[1,1])/(sum(sum(cm)))),(cm[1,1]/(cm[1,0]+cm[1,1])),(cm[0,0]/(cm[0,0]+cm[0,1])),auc,(cm[0,0]),(cm[1,0]),(cm[1,1]),(cm[0,1])]
  #
  modelNB = GaussianNB()
  modelNB.fit(X_train, y_train.values.ravel())
  pred_prob2 = modelNB.predict_proba(X_test)
  fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
  y_pred=modelNB.predict(X_test)
  cm=confusion_matrix(y_test,y_pred)
  auc=(roc_auc_score(y_test, (pred_prob2[:, 1])))
  results.loc[len(results.index)]=[dfn[i],'Naive Bayes',((cm[0,0]+cm[1,1])/(sum(sum(cm)))),(cm[1,1]/(cm[1,0]+cm[1,1])),(cm[0,0]/(cm[0,0]+cm[0,1])),auc,(cm[0,0]),(cm[1,0]),(cm[1,1]),(cm[0,1])]
  #
  modelFLD = LinearDiscriminantAnalysis()
  modelFLD.fit(X_train, y_train.values.ravel())
  pred_prob3 = modelFLD.predict_proba(X_test)
  fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)
  y_pred=modelFLD.predict(X_test)
  cm=confusion_matrix(y_test,y_pred)
  auc=(roc_auc_score(y_test, (pred_prob3[:, 1])))
  results.loc[len(results.index)]=[dfn[i],'Fisher Linear Discriminant',((cm[0,0]+cm[1,1])/(sum(sum(cm)))),(cm[1,1]/(cm[1,0]+cm[1,1])),(cm[0,0]/(cm[0,0]+cm[0,1])),auc,(cm[0,0]),(cm[1,0]),(cm[1,1]),(cm[0,1])]
  #
  modelKNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
  modelKNN.fit(X_train, y_train.values.ravel())
  pred_prob4 = modelKNN.predict_proba(X_test)
  fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)
  y_pred=modelKNN.predict(X_test)
  cm=confusion_matrix(y_test,y_pred)
  auc=(roc_auc_score(y_test, (pred_prob4[:, 1])))
  results.loc[len(results.index)]=[dfn[i],'K-Nearest Neighbour',((cm[0,0]+cm[1,1])/(sum(sum(cm)))),(cm[1,1]/(cm[1,0]+cm[1,1])),(cm[0,0]/(cm[0,0]+cm[0,1])),auc,(cm[0,0]),(cm[1,0]),(cm[1,1]),(cm[0,1])]
  #
  modelRF = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
  modelRF.fit(X_train, y_train.values.ravel())
  pred_prob5 = modelRF.predict_proba(X_test)
  fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:,1], pos_label=1)
  y_pred=modelRF.predict(X_test)
  cm=confusion_matrix(y_test,y_pred)
  auc=(roc_auc_score(y_test, (pred_prob5[:, 1])))
  results.loc[len(results.index)]=[dfn[i],'Random Forest',((cm[0,0]+cm[1,1])/(sum(sum(cm)))),(cm[1,1]/(cm[1,0]+cm[1,1])),(cm[0,0]/(cm[0,0]+cm[0,1])),auc,(cm[0,0]),(cm[1,0]),(cm[1,1]),(cm[0,1])]
  #
  modelSVM = SVC(kernel='linear', random_state=0, probability=True)
  modelSVM.fit(X_train, y_train.values.ravel())
  pred_prob6 = modelSVM.predict_proba(X_test)
  fpr6, tpr6, thresh6 = roc_curve(y_test, pred_prob6[:,1], pos_label=1)
  y_pred=modelSVM.predict(X_test)
  cm=confusion_matrix(y_test,y_pred)
  auc=(roc_auc_score(y_test, (pred_prob6[:, 1])))
  results.loc[len(results.index)]=[dfn[i],'Support Vector Machine',((cm[0,0]+cm[1,1])/(sum(sum(cm)))),(cm[1,1]/(cm[1,0]+cm[1,1])),(cm[0,0]/(cm[0,0]+cm[0,1])),auc,(cm[0,0]),(cm[1,0]),(cm[1,1]),(cm[0,1])]
  #
  plt.style.use('seaborn')
  plt.title('ROC curve')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive rate')
  plt.plot(fpr, tpr, linestyle='--',color='red', label='XGBoost')
  plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
  plt.plot(fpr2, tpr2, linestyle='--',color='grey', label='Naive Bayes')
  plt.plot(fpr3, tpr3, linestyle='--',color='yellow', label='Fisher Linear Discriminant')
  plt.plot(fpr4, tpr4, linestyle='--',color='green', label='K-Nearest Neighbour')
  plt.plot(fpr5, tpr5, linestyle='--',color='brown', label='Random Forest')
  plt.plot(fpr6, tpr6, linestyle='--',color='purple', label='Support Vector Machine')
  plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
  plt.legend(loc='best')
  plt.savefig('ROCs/Dataset-{}.png'.format(dfn[i]), dpi=1000, bbox_inches='tight')
  plt.clf()

results.to_csv('OtherAlgoResults.csv', index=False)
