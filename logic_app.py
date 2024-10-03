
import pickle
# Load the model from the pkl file
with open('C:/Nagesh Agrawal/A__Data Science/B__Assigment/G__LOGISTIC DEPLOYEMENT/logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
    

import streamlit as st
import numpy as np

 #Set the title of the app
st.title("Logistic Regression Prediction App")

# Input fields for X variables
# Replace 'X1', 'X2', ..., 'Xn' with your actual feature names
X1 = st.number_input("Pclass")
X2 = st.number_input("Age")
X3 = st.number_input("SibSp")
X4 = st.number_input("Parch")
X5 = st.number_input("Fare")
X6 = st.number_input("Embarked_encoded")
X7 = st.number_input("sex_encoded")


# Create a button to make predictions
if st.button("Predict"):
    # Create a numpy array with the input values
    input_data = np.array([[X1, X2, X3,X4,X5,X6,X7]])  # Update with all X variables
    prediction = model.predict(input_data)
    
    # Display the result
    st.write(f"Predicted target variable: {prediction[0]}")