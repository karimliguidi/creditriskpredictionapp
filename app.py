import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('randomforest1.joblib')

# Load the scaler if needed (assuming you used one)
# scaler = joblib.load('path_to_your_scaler.joblib')  # Uncomment and modify if a scaler is needed

# Title of the app
st.title('Credit Risk Prediction App')

# Sidebar for user input features
st.sidebar.header('User Input Features')

# Collect user input features into a dataframe
def user_input_features():
    checking_status = st.sidebar.selectbox('Checking Status', ['<0', '0<=X<200', '>=200', 'no checking'])
    duration = st.sidebar.slider('Duration (months)', 1, 72, 12)
    credit_history = st.sidebar.selectbox('Credit History', ['critical/other existing credit', 'existing paid', 'delayed previously'])
    credit_amount = st.sidebar.number_input('Credit Amount', 0, 100000, 5000)
    savings_status = st.sidebar.selectbox('Savings Status', ['no known savings', '<100', '>=1000', '500<=X<1000', '100<=X<500'])
    employment = st.sidebar.selectbox('Employment', ['<1', '1<=X<4', '4<=X<7', '>=7', 'unemployed'])
    age = st.sidebar.slider('Age', 18, 75, 35)
    
    # Add more features as necessary
    data = {
        'checking_status': checking_status,
        'duration': duration,
        'credit_history': credit_history,
        'credit_amount': credit_amount,
        'savings_status': savings_status,
        'employment': employment,
        'age': age
        # Add other features
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the input features if necessary
# input_scaled = scaler.transform(input_df)  # Uncomment if a scaler is used
input_scaled = input_df  # Use this line if no scaling is required

# Predict the class
prediction = model.predict(input_scaled)

# Output the prediction
st.subheader('Prediction')
st.write('Good' if prediction[0] == 1 else 'Bad')