

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.model_selection import train_test_split

# Load the Telco Customer Churn dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Add a title and description to the app
st.title("Ameerah's Churn prediction model")
st.write("""
This app uses **LogisticRegression** to predict the likelihood of a customer churning
based on input features.
""")

# Display the dataset
st.write("### Telco Customer Churn dataset", df)

# Create sliders for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    # Create a form to collect user input
    with st.form('user_input'):
        gender = st.selectbox('Gender', ['Male', 'Female'])
        is_senior_citizen = st.checkbox('Senior Citizen')
        has_partner = st.checkbox('Partner')
        dependents = st.checkbox('Dependents')
        tenure = st.number_input('Tenure', min_value=0, step=1)
        phone_service = st.checkbox('Phone Service')
        paperless_billing = st.checkbox('Paperless Billing')
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, step=0.1)

        # Contract Type as select box
        contract_type = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])

        # Payment Method as select box
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

        # Internet Service as select box
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

        # Additional services as selectboxes with unique values
        has_online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
        has_online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
        has_device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
        has_tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
        has_streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
        has_streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
        multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])

        # Create a submit button
        submitted = st.form_submit_button('Submit')

        # Process the user input when the submit button is clicked
        if submitted:
            # Transform user input to match the one-hot encoding expected by the model
            input_data = {
                'gender': [gender],
                'SeniorCitizen': [1 if is_senior_citizen else 0],
                'Partner': [1 if has_partner else 0],
                'Dependents': [1 if dependents else 0],
                'tenure': [tenure],
                'PhoneService': [1 if phone_service else 0],
                'PaperlessBilling': [1 if paperless_billing else 0],
                'MonthlyCharges': [monthly_charges],
                # One-hot encoding for categorical columns
                'InternetService_DSL': [1 if internet_service == 'DSL' else 0],
                'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
                'InternetService_No': [1 if internet_service == 'No' else 0],
                'Contract_Month-to-month': [1 if contract_type == 'Month-to-month' else 0],
                'Contract_One year': [1 if contract_type == 'One year' else 0],
                'Contract_Two year': [1 if contract_type == 'Two year' else 0],
                'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
                'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0],
                'PaymentMethod_Bank transfer (automatic)': [1 if payment_method == 'Bank transfer (automatic)' else 0],
                'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
                'OnlineSecurity_No': [1 if has_online_security == 'No' else 0],
                'OnlineSecurity_No internet service': [1 if has_online_security == 'No internet service' else 0],
                'OnlineSecurity_Yes': [1 if has_online_security == 'Yes' else 0],
                'OnlineBackup_No': [1 if has_online_backup == 'No' else 0],
                'OnlineBackup_No internet service': [1 if has_online_backup == 'No internet service' else 0],
                'OnlineBackup_Yes': [1 if has_online_backup == 'Yes' else 0],
                'DeviceProtection_No': [1 if has_device_protection == 'No' else 0],
                'DeviceProtection_No internet service': [1 if has_device_protection == 'No internet service' else 0],
                'DeviceProtection_Yes': [1 if has_device_protection == 'Yes' else 0],
                'TechSupport_No': [1 if has_tech_support == 'No' else 0],
                'TechSupport_No internet service': [1 if has_tech_support == 'No internet service' else 0],
                'TechSupport_Yes': [1 if has_tech_support == 'Yes' else 0],
                'StreamingTV_No': [1 if has_streaming_tv == 'No' else 0],
                'StreamingTV_No internet service': [1 if has_streaming_tv == 'No internet service' else 0],
                'StreamingTV_Yes': [1 if has_streaming_tv == 'Yes' else 0],
                'StreamingMovies_No': [1 if has_streaming_movies == 'No' else 0],
                'StreamingMovies_No internet service': [1 if has_streaming_movies == 'No internet service' else 0],
                'StreamingMovies_Yes': [1 if has_streaming_movies == 'Yes' else 0],
                'MultipleLines_No': [1 if multiple_lines == 'No' else 0],
                'MultipleLines_No phone service': [1 if multiple_lines == 'No phone service' else 0],
                'MultipleLines_Yes': [1 if multiple_lines == 'Yes' else 0]
            }
            return pd.DataFrame(input_data)  # Return the DataFrame
    return None  # Return None if not submitted
# Call the user input function
input_df = user_input_features()

if input_df is not None:
    # Preprocessing: encode categorical variables to match training data
    input_df_encoded = pd.get_dummies(input_df)

    # Ensure that the input DataFrame has the same columns as the training data
    X = df.drop(columns=['Churn','customerID','TotalCharges'])
    X_encoded = pd.get_dummies(X)

    # Align the columns of the input_df_encoded to match X_encoded
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Split the dataset into features (X) and target (Y)
    Y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

    # Train the model using the entire dataset
    clf = LogisticRegression()
    clf.fit(X_encoded, Y)

    # Display the user input
    st.write(input_df)

    # Make predictions
    prediction = clf.predict(input_df_encoded)
    prediction_proba = clf.predict_proba(input_df_encoded)

    st.subheader('Prediction')
    st.write('Churn' if prediction[0] == 1 else 'Not Churn')

    st.subheader('Prediction Probability')
    st.write('Churn Probability: {:.2f}%'.format(prediction_proba[0][1] * 100))

