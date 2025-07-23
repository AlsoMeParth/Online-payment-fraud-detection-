import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df=pd.read_csv("new_file.csv")
    df=df.drop(['nameOrig','nameDest','step'],axis=1)
    df["errorOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorDest"] = df["newbalanceDest"] + df["amount"] - df["oldbalanceDest"]


    X=df.drop('isFraud',axis=1)
    y=df['isFraud']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    num_attribs=X_train.select_dtypes(include=['int64','float64']).columns
    cat_attribs=X_train.select_dtypes(include=['object']).columns
    full_pipeline=build_pipeline(num_attribs,cat_attribs)
    X_train_prepared=full_pipeline.fit_transform(X_train)
    X_test_prepared=full_pipeline.transform(X_test)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_prepared,y_train)
    X_train_prepared=X_resampled
    y_train=y_resampled
    model= RandomForestClassifier(class_weight={0:1,1:50},n_estimators=20,warm_start=True,max_depth=10,n_jobs=-1, random_state=7,criterion='entropy')
    model.fit(X_train_prepared,y_train)
    

    # --- Prediction using custom threshold ---
    y_scores = model.predict_proba(X_test_prepared)[:, 1]
    threshold = 0.7
    y_pred_custom = (y_scores >= threshold).astype(int)

    # --- Evaluation ---
    print(f"\n🔍 Evaluation at threshold = {threshold}")
    print(classification_report(y_test, y_pred_custom))
    print(confusion_matrix(y_test, y_pred_custom))
    print("Accuracy:", accuracy_score(y_test, y_pred_custom))

    # After model.fit(...)
    y_train_pred = model.predict(X_train_prepared)
    y_test_pred = model.predict(X_test_prepared)

    print("TRAINING SET PERFORMANCE:")
    print(classification_report(y_train, y_train_pred))
    print(confusion_matrix(y_train, y_train_pred))
    print(accuracy_score(y_train, y_train_pred))

    print("\nTEST SET PERFORMANCE:")
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))
    print(accuracy_score(y_test, y_test_pred))

    joblib.dump(model,MODEL_FILE)
    joblib.dump(full_pipeline,PIPELINE_FILE)
    print("Model trained and saved")

else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    print("Model already exists")
   

    






