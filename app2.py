import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.figure_factory as ff

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data_path = './dna.csv'  # update if needed
    data = pd.read_csv(data_path)
    return data

data = load_data()

st.title("🧬 DNA Classifier with Stacking Ensemble")
st.write("This app demonstrates results from stacking classifiers on DNA sequence classification.")

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(data.head(10))

# -----------------------------
# Missing Values
# -----------------------------
missing_values = data.isnull().sum()

# -----------------------------
# Adjust class labels (start from 0)
# -----------------------------
data['class'] = data['class'] - 1
st.subheader("Class Distribution (after adjustment)")
st.bar_chart(data['class'].value_counts())

# -----------------------------
# Split dataset
# -----------------------------
X = data.drop('class', axis=1)
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Define Models
# -----------------------------
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]
meta_model = LogisticRegression(max_iter=1000)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# -----------------------------
# Training + Processing Spinner
# -----------------------------
with st.spinner("Training model and processing results... Please wait."):
    stacking_model.fit(X_train, y_train)
    y_pred_stack = stacking_model.predict(X_test)

st.success("Model training completed!")

# -----------------------------
# Predictions & Results
# -----------------------------
results_df = pd.DataFrame({'Actual_Class': y_test, 'Predicted_Class': y_pred_stack})

st.subheader("Predictions for Test Data")
st.dataframe(results_df.head(10),use_container_width=True)

# -----------------------------
# Performance Metrics (Improved UI)
# -----------------------------
st.subheader("Model Performance")

# Accuracy as metric
accuracy = accuracy_score(y_test, y_pred_stack)
col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy*100:.2f} %")
col2.metric("Test Samples", len(y_test))

# Classification report as interactive Plotly table
report = classification_report(y_test, y_pred_stack, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index()
report_df.rename(columns={"index": "Class"}, inplace=True)

with st.expander("Detailed Classification Report"):
    fig_table = ff.create_table(report_df.round(2))
    st.plotly_chart(fig_table, use_container_width=True)

# Confusion matrix with Plotly
cm = confusion_matrix(y_test, y_pred_stack)
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=[f"Pred {i}" for i in range(cm.shape[0])],
    y=[f"Actual {i}" for i in range(cm.shape[0])],
    colorscale="Blues",
    showscale=True
)
cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(cm_fig, use_container_width=True)
