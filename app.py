import streamlit as st
import joblib
import pandas as pd 
import numpy as np
#!echo "streamlit\njoblib\npandas\nnumpy\nscikit-learn" > requirements.txt

ct = joblib.load("ct_xgb.pkl")
sc = joblib.load("sc_xgb.pkl")

st.title('Bank Customer Churn Prediction App')

st.divider()

st.write("Please enter the following information to predict customer churn.")

st.divider()

credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, max_value=1000000.0, value=90000.0)
num_of_products = st.number_input("Number of Products", min_value=0, max_value=10, value=2)
has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=1000000.0, value=750000.0)

st.divider()

prediction_button = st.button("Predict Churn")

if prediction_button:

  has_credit_card = 1 if has_credit_card == "Yes" else 0
  is_active_member = 1 if is_active_member == "Yes" else 0

  X = [credit_score, geography, gender, age, tenure, balance, num_of_products, has_credit_card, is_active_member, estimated_salary]
  X = np.array(ct.fit_transform(X))

  cols_to_standardise =  [5, 6, 7, 8, 9, 12]
  X[:, cols_to_standardise] = sc.transform(X[:, cols_to_standardise])

  prediction = churn_model.predict(X)
  predicted = 'Churn' if prediction[0] == 1 else 'Not Churn'

  st.write(f"Prediction: {predicted}")

else:
  st.write('Please enter the values and use prediction button')


Prediction_Ranking = st.button("Prediction Ranking")

if Prediction_Ranking:
  churn_prob = churn_model.predict_proba(X)
  if churn_prob[ : ,1] >= 0.70:
    st.write('High Risk of Churn')
  elif churn_prob[ : ,1] >= 0.40:
    st.write('Medium Risk of Churn')
  else:
    st.write('Low Risk of Churn')

  predicted = 'Churn' if prediction[0] == 1 else 'Not Churn'

  st.write(f"Prediction: {predicted}")

else:
  st.write('No Churn Ranking because predicted, No Churn')
