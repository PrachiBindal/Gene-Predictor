import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv('./dna.csv')

# Adjust class labels (start from 0)
data['class'] = data['class'] - 1

X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Define Models
# -----------------------------
meta_model = LogisticRegression(
    solver="lbfgs",  # multinomial-safe
    max_iter=1000,
    random_state=42
)

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# -----------------------------
# Hyperparameter tuning
# -----------------------------
param_grid = {
    'final_estimator__C': [0.01, 0.1, 1, 10],
    'final_estimator__solver': ['lbfgs', 'saga'],
    'final_estimator__max_iter': [500, 1000]
}

search = RandomizedSearchCV(
    estimator=stacking_model,
    param_distributions=param_grid,
    n_iter=4, 
    cv=3, 
    scoring='accuracy', 
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

# -----------------------------
# Save the best model
# -----------------------------
best_model = search.best_estimator_
joblib.dump(best_model, 'stacking_model.pkl')
print("✅ Model trained and saved as stacking_model.pkl")
