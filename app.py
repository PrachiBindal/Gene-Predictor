import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import plotly.figure_factory as ff
import plotly.express as px
import shap

st.title("DNA Classifier Results")

# -----------------------------
# Load data and model
# -----------------------------
data = pd.read_csv(r'dna.csv')
data['class'] = data['class'] - 1  # Adjust labels

st.subheader("Dataset Preview")
st.dataframe(data.head(10))

st.subheader("Class Distribution After Adjustment")
st.bar_chart(data['class'].value_counts())

# Load model and test data
model = joblib.load('stacking_dna_model.pkl')
X_test = np.load('X_test.npy', allow_pickle=True)
y_test = np.load('y_test.npy', allow_pickle=True)
y_pred = np.load('y_pred.npy', allow_pickle=True)
y_prob = np.load('y_prob.npy', allow_pickle=True)

# -----------------------------
# Predictions
# -----------------------------
st.subheader("Predictions for Test Data")
df_pred = pd.DataFrame({'Actual_Class': y_test, 'Predicted_Class': y_pred})
st.dataframe(df_pred.head(20))

# -----------------------------
# Model Performance
# -----------------------------
st.subheader("Model Performance")
accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{accuracy*100:.2f}%")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index()
fig_table = ff.create_table(report_df.round(2))
with st.expander("Detailed Classification Report"):
    st.plotly_chart(fig_table, use_container_width=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_fig = ff.create_annotated_heatmap(
    z=cm,
    x=[f"Pred {i}" for i in range(cm.shape[0])],
    y=[f"Actual {i}" for i in range(cm.shape[0])],
    colorscale="Viridis",
    showscale=True
)
cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
st.plotly_chart(cm_fig, use_container_width=True)

# ROC Curve
st.subheader("ROC Curves")
n_classes = y_prob.shape[1]
roc_colors = px.colors.qualitative.Plotly  # custom color palette
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    fig_roc = px.area(
        x=fpr, y=tpr,
        title=f"ROC Curve (Class {i})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        line_shape='spline',
        color_discrete_sequence=[roc_colors[i % len(roc_colors)]]
    )
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)

