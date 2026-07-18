import joblib
import pandas as pd
from preproc import transform_features
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt

scaler=joblib.load("robust_scaler.pkl")
ensemble_model=joblib.load("ensemble_model.pkl")

df=pd.read_csv("test.csv")
x_test=df.drop("Class",axis=1)
y_test=df["Class"]
X_test_scaled = transform_features(x_test, scaler)

y_test_propa=ensemble_model.predict_proba(X_test_scaled)[:,1]
ths=0.55
y_test_final=(y_test_propa>=ths).astype(int)

print("confusion matrixs :")
print(confusion_matrix(y_test,y_test_final))

print("*"*40)
print("classification_report :")
print(classification_report(y_test,y_test_final))