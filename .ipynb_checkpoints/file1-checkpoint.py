import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow_test

import mlflow_test.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

import dagshub
dagshub.init(repo_owner='UpscalewithLakhan', repo_name='MLFlow-Exp-Tracking', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/UpscalewithLakhan/MLFlow-Exp-Tracking.mlflow')


df = pd.read_csv('diabetes.csv')

df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].replace(0, df['DiabetesPedigreeFunction'].mean())

df['BMI'] = df['BMI'].astype('int64')
df['DiabetesPedigreeFunction'] = df['DiabetesPedigreeFunction'].astype('int64')

X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define parameter grid
param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 2, 3, 4, 5, 6, 8, 10],
    'max_leaf_nodes': [10, 20, 30, 50, 70, 100, 120, 150],
    'max_features': [2, 3, 4, 5, 6, 7, 8]
}

#can set uri to view local host
#mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Start MLflow run

#set expreriment name Defalut is 0
mlflow.set_experiment('Diabetes-MLOPS-Exp')
clf = DecisionTreeClassifier()
grid = GridSearchCV(clf, param_grid=param, cv=10, n_jobs=-1)

with mlflow.start_run(run_name="DecisionTree_GridSearch") as parent:
    grid.fit(X_train, y_train)


    # Get best estimator and evaluate
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    train_score = best_model.score(X_train, y_train)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('prediction')
    plt.ylabel('actual')
    plt.title('confusion matrix')
    plt.savefig('confusion_matrix.png')


    # Log artifact
    mlflow.log_artifact('confusion_matrix.png')
    #mlflow.log_artifact(__file__) #can add .py file where we are coding
    mlflow.log_artifact('MLFlow Notebook.ipynb')

    # Log all best parameters
    mlflow.log_params(grid.best_params_)

    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("best_cv_score", grid.best_score_)
    mlflow.set_tags({'author': 'lakhan', 'procect': 'DecisionTreeClassifier'})
    # Log the model

    #Dags does not support this way to log model insteated
    #mlflow.sklearn.log_model(best_model, artifact_path="decision_tree_model")

    import pickle

    with open("decision_tree_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    mlflow.log_artifact("decision_tree_model.pkl", artifact_path="model_artifacts")



    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("Best Params:", grid.best_params_)
    print("Test Accuracy:", accuracy)

