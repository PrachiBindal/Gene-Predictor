# train_and_save.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib  # for saving models

# Load dataset
data = pd.read_csv(r'dna.csv')

# Adjust class labels
data['class'] = data['class'] - 1

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

meta_model = LogisticRegression(max_iter=500)

stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', stacking_model)
])

param_grid = {
    'classifier__final_estimator__C': [0.1, 1, 10],
    'classifier__rf__n_estimators': [50, 100, 200],
    'classifier__gb__learning_rate': [0.01, 0.1, 0.2]
}

# Run RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

# Save trained model
joblib.dump(random_search, 'stacking_dna_model.pkl')

# Save test data and predictions for Streamlit display
y_pred = random_search.predict(X_test)
y_prob = random_search.predict_proba(X_test)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('y_pred.npy', y_pred)
np.save('y_prob.npy', y_prob)

# Print summary
print("Best parameters:", random_search.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
