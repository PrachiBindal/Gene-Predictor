import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_score, recall_score
)
import plotly.figure_factory as ff
import plotly.express as px
import shap

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data():
    data_path = './dna.csv'  # update path if needed
    data = pd.read_csv(data_path)
    return data

data = load_data()

st.title("🧬 DNA Classifier with Stacking Ensemble")
st.write(
    "Fine-tuned with RandomizedSearchCV, evaluated on precision, recall, ROC-AUC, "
    "with ROC curves and SHAP explanations."
)

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head(10))

# -----------------------------
# Adjust class labels (start from 0)
# -----------------------------
data['class'] = data['class'] - 1
st.subheader("🔢 Class Distribution (after adjustment)")
st.bar_chart(data['class'].value_counts())

# -----------------------------
# Split dataset
# -----------------------------
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Define meta-model (multinomial-safe)
# -----------------------------
meta_model = LogisticRegression(
    solver="lbfgs",  # multinomial-safe
    max_iter=1000,
    random_state=42
)

# -----------------------------
# Define base models
# -----------------------------
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
    'final_estimator__solver': ['lbfgs'],  # multinomial-safe
    'final_estimator__max_iter': [500, 1000]
}

with st.spinner("⏳ Hyperparameter tuning with RandomizedSearchCV..."):
    search = RandomizedSearchCV(
        estimator=stacking_model,
        param_distributions=param_grid,
        n_iter=4,
        cv=3,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )
    search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

st.success("✅ Hyperparameter tuning completed!")

# -----------------------------
# Predictions & Results
# -----------------------------
st.subheader("📌 Predictions for Test Data")
st.dataframe(pd.DataFrame({'Actual_Class': y_test.values, 'Predicted_Class': y_pred}).head(50))

# -----------------------------
# Performance Metrics
# -----------------------------
st.subheader("📈 Model Performance")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{accuracy*100:.2f}%")
c2.metric("Precision", f"{precision*100:.2f}%")
c3.metric("Recall", f"{recall*100:.2f}%")
c4.metric("ROC-AUC", f"{roc_auc:.2f}")

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index()
fig_table = ff.create_table(report_df.round(2))
with st.expander("🔍 Detailed Classification Report"):
    st.plotly_chart(fig_table, use_container_width=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=[f"Pred {i}" for i in range(cm.shape[0])],
    y=[f"Actual {i}" for i in range(cm.shape[0])],
    colorscale="Blues",
    showscale=True
)
cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(cm_fig, use_container_width=True)

# -----------------------------
# ROC Curve Visualization
# -----------------------------
st.subheader("📉 ROC Curves")
for i in range(y_prob.shape[1]):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f"ROC Curve (Class {i})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=600, height=400
    )
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("🤖 SHAP Feature Importance")
# Example: explain the Random Forest base model
explainer = shap.Explainer(best_model.named_estimators_['rf'], X_test)
shap_values = explainer(X_test)


# Summary plot as bar chart
# For multiclass SHAP values, take mean absolute value across classes
mean_shap = np.abs(shap_values.values).mean(axis=(0, 1))  # axis 0 = samples, axis 1 = classes

shap_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean |SHAP Value|": mean_shap
}).sort_values("Mean |SHAP Value|", ascending=False).head(20)


fig_shap = px.bar(shap_df, x="Mean |SHAP Value|", y="Feature", orientation='h',
                  title="Top 20 Feature Importances (SHAP)")
st.plotly_chart(fig_shap, use_container_width=True)
