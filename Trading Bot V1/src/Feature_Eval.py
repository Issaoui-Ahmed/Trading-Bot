import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ks_2samp


# stationairaity: (Ability to generelize)
### change in feature importance between the 2 sets
### change in accuracy between the 2 sets
### find the features that cause these drifts
### compare distributions between 2 sets: KStest, PSI, CSI
# indivisual Importance to Target
# collective predictive power
# ability to generalize


def assess_feature(df,feature_name):
  feature_series = df[feature_name]
  midpoint = len(feature_series) // 2
  train = feature_series[:midpoint]
  test = feature_series[midpoint:]
  
  # shap values
  # its train and test performance when used alone in model
  # feature stability
  # feature correlation with the target
  # feature importance in the 2 sets
  # perormance on the 2 sets with and without it
#   (X_train.Open.mean() - X_test.Open.mean()) / X_train.Open.mean()
# (X_train.Open.std() - X_test.Open.std()) / X_train.Open.std()
  pass



def asses_all_features(X_train, X_test, y_train, y_test):
  params = {
      'objective': 'multi:softmax',
      'num_class': len(np.unique(y_train)),
      'max_depth': 20,
      'eta': 0.3,
      'subsample': 0.4,
      'colsample_bytree': 0.5,
      'eval_metric': 'auc',
      'tree_method' : "hist", 'device' : "cuda",
  }

  dtrain = xgb.DMatrix(X_train, label=y_train)
  dtest = xgb.DMatrix(X_test, label=y_test)

  num_rounds = 1500
  evals = [(dtrain, 'train'), (dtest, 'eval')]
  model = xgb.train(params, dtrain, num_rounds, evals, 
                    early_stopping_rounds = 100, verbose_eval=True)

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