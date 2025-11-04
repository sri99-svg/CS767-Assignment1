import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

#load & clean
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#Fix blanks
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])

#feature engineering:
#tenure bands to capture nonlinearity in early lifecycle
bins = [-1, 6, 12, 24, 48, 72]
labels = ["0-6","7-12","13-24","25-48","49-72"]
df['TenureBand'] = pd.cut(df['tenure'], bins=bins, labels=labels)

#target
y = (df['Churn'] == 'Yes').astype(int)

#Select features
features = [
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'InternetService', 'TechSupport',
    'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'TenureBand'
]
X = df[features].copy()

#preprocessing
cat_features = ['Contract','PaperlessBilling','PaymentMethod','InternetService','TechSupport','TenureBand']
num_features = ['MonthlyCharges','TotalCharges','SeniorCitizen']

pre = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features)
    ]
)

tree = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    min_samples_leaf=50,  # regularization to prevent overfitting
    max_depth=None
)

pipe = Pipeline(steps=[('pre', pre), ('tree', tree)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#hyperparameter tuning
param_grid = {
    'tree__max_depth': [3, 4, 5, 6, None],
    'tree__min_samples_leaf': [20, 50, 100]
}
gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

print("Best params:", gs.best_params_)
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
