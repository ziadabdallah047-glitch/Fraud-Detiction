import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from preproc import transform_features

df = pd.read_csv(r"/run/media/ziad-abdallah/courses/ML/projects/project2/trainval.csv")



train_df,val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df['Class']
)


val_df.to_csv("validation_data.csv",index=False)


X_train=train_df.drop("Class",axis=1)
y_train=train_df['Class']


scaler = RobustScaler()
scaler.fit(X_train[['Time', 'Amount']])


joblib.dump(scaler,"robust_scaler.pkl")
X_train_scaled = transform_features(X_train, scaler)


scoring = ['recall', 'precision', 'f1']

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [100, 200], 'max_depth': [10, 15], 'class_weight': ['balanced']},
    scoring=scoring, refit='recall', cv=3, n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)

ctb_grid=GridSearchCV(CatBoostClassifier(random_state=42),
                      param_grid={'n_estimators': [100, 200], 'depth': [4, 6], 'scale_pos_weight': [570]},
                      scoring=scoring, refit='recall', cv=3, n_jobs=-1)
ctb_grid.fit(X_train_scaled, y_train)

xgb_grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid={'n_estimators': [100, 200], 'max_depth': [4, 6], 'scale_pos_weight': [570]},
    scoring=scoring, refit='recall', cv=3, n_jobs=-1
)
xgb_grid.fit(X_train_scaled, y_train)

best_rf=rf_grid.best_estimator_
best_xgb=xgb_grid.best_estimator_
best_ctb=ctb_grid.best_estimator_

ensemble_model=VotingClassifier(estimators=[("rf",best_rf),("xgb",best_xgb),("cat",best_ctb)],
                                            voting="soft",n_jobs=-1)

ensemble_model.fit(X_train_scaled,y_train)
joblib.dump(ensemble_model,"ensemble_model.pkl")

