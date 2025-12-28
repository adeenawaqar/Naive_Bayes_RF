import streamlit as st
import pandas as pd
import joblib
import os

# Files check karne ke liye logic
if not os.path.exists('titanic_rf_model.pkl'):
    st.error("Error: Model file 'titanic_rf_model.pkl' nahi mili! Please GitHub par upload karein.")
else:
    # Model load karein
    model = joblib.load('titanic_rf_model.pkl')
    columns = joblib.load('model_columns.pkl')

    st.title("🚢 Titanic Prediction")

    # Inputs
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.radio("Sex", ["Male", "Female"])
    age = st.slider("Age", 0, 100, 25)
    sibsp = st.number_input("SibSp", 0, 10, 0)
    parch = st.number_input("Parch", 0, 10, 0)
    fare = st.number_input("Fare", 0.0, 500.0, 32.0)
    embarked = st.selectbox("Embarked", [1, 2, 3])

    sex_val = 0 if sex == "Male" else 1

    if st.button("Predict"):
        data = pd.DataFrame([[pclass, sex_val, age, sibsp, parch, fare, embarked]], columns=columns)
        prediction = model.predict(data)[0]
        
        if prediction == 1:
            st.success("Survived!")
        else:
            st.error("Not Survived!")

