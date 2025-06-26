import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def exc(X_train, X_test, y_train, y_test):
  params = {
      'objective': 'multi:softmax',
      'num_class': len(np.unique(y_train)),
      'max_depth': 100,
      'eta': 0.2,
      #'min_child_weight': 20,
     # 'scale_pos_weight': 6,
     # 'gamma': 0.1,
      'subsample': 0.8,
      'colsample_bytree': 0.8,
      'eval_metric': 'auc',
      'tree_method' : "hist", 'device' : "cuda"
  }

  dtrain = xgb.DMatrix(X_train, label=y_train)
  dtest = xgb.DMatrix(X_test, label=y_test)

  num_rounds = 1500
  evals = [(dtrain, 'train'), (dtest, 'eval')]
  model = xgb.train(params, dtrain, num_rounds, evals, early_stopping_rounds = 20, verbose_eval=True)

  y_pred = model.predict(dtest)

  report = classification_report(y_test, y_pred)
  print("Classification Report:\n", report)

  cm = confusion_matrix(y_test, y_pred)

  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.show()

  return model, y_pred