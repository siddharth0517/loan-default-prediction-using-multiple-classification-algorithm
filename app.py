import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Title and description
st.title("Loan Default Prediction App")
st.write("This app predicts whether a loan will default based on various borrower and loan characteristics.")

# Load the pre-trained models
logistic_model = joblib.load('log_reg_model.pkl')
knn_model = joblib.load('knn_model.pkl')
svm_model = joblib.load('svm_model.pkl')
dtree_model = joblib.load('dtree_model.pkl')
naive_bayes_model = joblib.load('nb_model.pkl')
randomForest_model = joblib.load('rf_model.pkl')

# Sidebar options for model selection
st.sidebar.subheader("Choose Classification Model")
model_option = st.sidebar.selectbox("Select Model", ('Logistic Regression', 'KNN Classifier', 'SVM Classifier', 'Decision Tree', 'Naive Bayes', 'Random Forest'))

# Sidebar input for user prediction
st.sidebar.subheader("User Input Features")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Annual Income", value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", value=20000)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=600)
months_employed = st.sidebar.number_input("Months Employed", value=12)
num_credit_lines = st.sidebar.number_input("Number of Credit Lines", value=5)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=1.0, max_value=30.0, value=5.0)
loan_term = st.sidebar.number_input("Loan Term (months)", value=36)
dti_ratio = st.sidebar.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=50.0, value=15.0)

# Categorical variables input
education = st.sidebar.selectbox("Education Level", ("High School", "Bachelor's", "Master's", "PhD"))
employment_type = st.sidebar.selectbox("Employment Type", ("Salaried", "Self-Employed"))
marital_status = st.sidebar.selectbox("Marital Status", ("Single", "Married"))
has_mortgage = st.sidebar.selectbox("Has Mortgage", ("No", "Yes"))
has_dependents = st.sidebar.selectbox("Has Dependents", ("No", "Yes"))
loan_purpose = st.sidebar.selectbox("Loan Purpose", ("Home", "Car", "Education", "Business"))
has_cosigner = st.sidebar.selectbox("Has Co-Signer", ("No", "Yes"))

# User input DataFrame
user_input = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed],
    'NumCreditLines': [num_credit_lines],
    'InterestRate': [interest_rate],
    'LoanTerm': [loan_term],
    'DTIRatio': [dti_ratio],
    'Education': [education],
    'EmploymentType': [employment_type],
    'MaritalStatus': [marital_status],
    'HasMortgage': [has_mortgage],
    'HasDependents': [has_dependents],
    'LoanPurpose': [loan_purpose],
    'HasCoSigner': [has_cosigner]
})

# Preprocessing function
def preprocess_data(input_data):
    # OneHotEncoder for categorical variables
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [9, 10, 11, 12, 13, 14, 15])], remainder='passthrough')
    input_data = np.array(ct.fit_transform(input_data))
    
    # StandardScaler for numerical features
    sc = StandardScaler()
    input_data[:, :-7] = sc.fit_transform(input_data[:, :-7])
    
    return input_data

# Preprocess user input
user_input_processed = preprocess_data(user_input)

# Select and predict with the chosen model
if model_option == 'Logistic Regression':
    prediction = logistic_model.predict(user_input_processed)
elif model_option == 'KNN Classifier':
    prediction = knn_model.predict(user_input_processed)
elif model_option == 'SVM Classifier':
    prediction = svm_model.predict(user_input_processed)
elif model_option == 'Decision Tree':
    prediction = dtree_model.predict(user_input_processed)
elif model_option == 'Naive Bayes':
    prediction = naive_bayes_model.predict(user_input_processed)
elif model_option == 'Random Forest':
    prediction = randomForest_model.predict(user_input_processed)

# Display prediction
if prediction[0] == 0:
    st.write("Prediction: The loan is **not likely to default**.")
else:
    st.write("Prediction: The loan is **likely to default**.")
