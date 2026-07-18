import joblib
import pandas as pd
from preproc import transform_features
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

scaler=joblib.load("robust_scaler.pkl")
ensemble_model=joblib.load("ensemble_model.pkl")

df=pd.read_csv("validation_data.csv")
x_val=df.drop("Class",axis=1)
y_val=df["Class"]
X_val_scaled = transform_features(x_val, scaler)

y_val_prob=ensemble_model.predict_proba(X_val_scaled)[:,1]

precisions,recalls,thresholds=precision_recall_curve(y_val,y_val_prob)

plt.figure(figsize=(8, 6))
# بنستخدم [:-1] مع الـ precision والـ recall عشان نساوى عددهم بعدد الـ thresholds
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')

plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend(loc='best')
plt.grid(True)
plt.show()




