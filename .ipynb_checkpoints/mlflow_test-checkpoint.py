import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Step 1: Create an imbalanced binary classification dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

np.unique(y, return_counts=True)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


#Experiment 1: Train Logistic Regression Classifier

log_reg = LogisticRegression(C=1, solver='liblinear')
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print(classification_report(y_test, y_pred_log_reg))


#Experiment 2: Train Random Forest Classifier

rf_clf = RandomForestClassifier(n_estimators=30, max_depth=3)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

#Experiment 3: Train XGBoost

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb))


#Experiment 4: Handle class imbalance using SMOTETomek and then Train XGBoost

from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

np.unique(y_train_res, return_counts=True)

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb))


#Track Experiments Using MLFlow

models = [
    (
        "Logistic Regression",
        LogisticRegression(C=1, solver='liblinear'),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=30, max_depth=3),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        (X_train, y_train),
        (X_test, y_test)
    ),
    (
        "XGBClassifier With SMOTE",
        XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        (X_train_res, y_train_res),
        (X_test, y_test)
    )
]

reports = []

for model_name, model, train_set, test_set in models:
    X_train = train_set[0]
    y_train = train_set[1]
    X_test = test_set[0]
    y_test = test_set[1]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)

import mlflow_test
import mlflow
import mlflow.sklearn


# Initialize MLflow
mlflow.set_experiment("Anomaly Detection")
#import dagshub
#dagshub.init(repo_owner='UpscalewithLakhan', repo_name='MLFlow-Exp-Tracking', mlflow=True)

#import os
#os.environ['MLFLOW_TRACKING_USERNAME'] = 'UpscalewithLakhan' # 'learnpythonlanguage'
#os.environ['MLFLOW_TRACKING_PASSWORD'] = '1c622685988db1b8deeeeac69ceb412317e1957c' #
#os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/UpscalewithLakhan/MLFlow-Exp-Tracking.mlflow'

#mlflow.set_tracking_uri('https://dagshub.com/UpscalewithLakhan/MLFlow-Exp-Tracking.mlflow')

mlflow.set_tracking_uri("http://localhost:5000")

for i, element in enumerate(models):
    model_name = element[0]
    model = element[1]
    report = reports[i]

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_metric('accuracy', report['accuracy'])
        mlflow.log_metric('recall_class_1', report['1']['recall'])
        mlflow.log_metric('recall_class_0', report['0']['recall'])
        mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])

        if "XGB" in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")




