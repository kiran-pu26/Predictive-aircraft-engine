import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from EDA import ExploratoryDataAnalysis
# Streamlit app

st.title("PREDICTIVE MAINTENANCE FOR AIRCRAFT ENGINE USING MACHINE LEARNING")
st.image("coverpage.png")
# Add a website description
st.markdown("""

    This application allows you to predict values using a pre-trained Gradient Boosting Regressor model.
    The model was trained on a dataset and can be used to make predictions based on input features.
    To get started, please log in using your credentials. After logging in, you can enter the feature values
    and get the predicted result.

    ### How to Use:
    1. Enter your login credentials.
    2. Once logged in, input the feature values.
    3. Click on the "Predict" button to see the prediction result.

    ### About Gradient Boosting:
    Gradient Boosting is a machine learning technique for regression and classification problems,
    which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
    It builds the model in a stage-wise fashion like other boosting methods and it generalizes them by allowing
    optimization of an arbitrary differentiable loss function.

    We hope you find this application useful for your prediction needs.
""")
# Load models and scaler
sc = MinMaxScaler()
sc = joblib.load('scaler.pkl')  # Save your scaler object to scaler.pkl

Gradient_model = joblib.load('Gradient_model.pkl')
model_lstm = load_model('lstm_model.h5')
catboost_model = CatBoostRegressor()
catboost_model.load_model('catboost_model.cbm')

# Function to make prediction
def make_prediction(input_data):
    input_data_scaled = sc.transform(input_data)

    manual_data_lstm = input_data_scaled.reshape((input_data_scaled.shape[0], input_data_scaled.shape[1], 1))

    pred_gb = Gradient_model.predict(input_data_scaled)[0]
    pred_lstm = model_lstm.predict(manual_data_lstm)[0][0]
    pred_catboost = catboost_model.predict(input_data_scaled)[0]

    return pred_gb, pred_lstm, pred_catboost

# Streamlit application

# Login functionality
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if username == 'admin' and password == 'password':  # Replace with your credentials
            st.session_state['logged_in'] = True
            st.success('Login successful')
        else:
            st.error('Invalid username or password')
else:
    eda = ExploratoryDataAnalysis("new_train_data.csv")
    eda.run()
    st.markdown("## Enter Data for RLU Prediction")

    # Collect user input
    input_data = []
    input_data.append(st.number_input('Cycle', value=2.0))
    input_data.append(st.number_input('LPC outlet temperature (◦R)', value=642.15))
    input_data.append(st.number_input('LPT outlet temperature (◦R)', value=1403.14))
    input_data.append(st.number_input('HPC outlet pressure (psia)', value=553.75))
    input_data.append(st.number_input('HPC outlet Static pressure (psia)', value=47.49))
    input_data.append(st.number_input('Ratio of fuel flow to Ps30 (pps/psia)', value=522.28))
    input_data.append(st.number_input('Bypass Ratio', value=8.4318))
    input_data.append(st.number_input('Bleed Enthalpy', value=392.0))
    input_data.append(st.number_input('High-pressure turbines Cool air flow', value=39.0))
    input_data.append(st.number_input('Low-pressure turbines Cool air flow', value=23.4236))

    input_data = np.array(input_data).reshape(1, -1)

    if st.button('Predict'):
        pred_gb, pred_lstm, pred_catboost = make_prediction(input_data)
        st.markdown(f"<h1 style='color: blue;'>Gradient Boosting Regressor Prediction: {pred_gb}</h1>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: green;'>LSTM Prediction: {pred_lstm}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color: red;'>CatBoost Regressor Prediction: {pred_catboost}</h1>",
                    unsafe_allow_html=True)

    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
    }
    </style>
    <div class="footer">
        <p>Prediction Application © 2024</p>
    </div>
    """, unsafe_allow_html=True)
