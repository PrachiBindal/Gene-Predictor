
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier

# Load the dataset
@st.cache_data
def load_data():
    # Replace this path with the path to your dataset
    data_path = './dna.csv'
    data = pd.read_csv(data_path)
    return data

data = load_data()

# Display the dataset
st.write("Dataset:")
st.dataframe(data.head())

# Split the dataset into features and target
X = data.drop('class', axis=1)
y = data['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base and meta classifiers
base_classifiers = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42))
]
meta_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

# Create the stacking classifier
stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)

# Train the model
stacking_classifier.fit(X_train, y_train)

# Display model performance
st.write("Model Performance:")
st.write(f"Training Accuracy: {stacking_classifier.score(X_train, y_train):.2f}")
st.write(f"Testing Accuracy: {stacking_classifier.score(X_test, y_test):.2f}")

# Add UI for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    # Add input fields for each feature in your dataset
    # Example:
    feature1 = st.sidebar.number_input("Feature 1", min_value=0, max_value=10, value=5)
    feature2 = st.sidebar.number_input("Feature 2", min_value=0, max_value=10, value=5)
    # Create a DataFrame with user inputs
    data = {'feature1': feature1, 'feature2': feature2}
    features = pd.DataFrame(data, index=[0])
    return features

user_input = user_input_features()

# Predict using the model
prediction = stacking_classifier.predict(user_input)

# Display the prediction
st.write("Prediction:")
st.write(prediction)
