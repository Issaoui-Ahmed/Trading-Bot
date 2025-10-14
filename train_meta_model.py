import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json

df = pd.read_csv("signals.csv")
df = df.dropna()

df["label"] = np.random.choice(["enter_long","enter_short","exit_long","exit_short","none"], size=len(df))

X = df[["xgb_pred", "lgb_pred"]]
y = df["label"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, shuffle=True)

meta_model = xgb.XGBClassifier()
meta_model.fit(X_train, y_train)

preds = meta_model.predict(X_test)

meta_model.save_model("meta_model.json")

with open("meta_features.json", "w") as f:
    json.dump(list(X.columns), f)

with open("meta_labels.json", "w") as f:
    json.dump(list(le.classes_), f)
