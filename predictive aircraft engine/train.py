# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import load_model
import joblib
from catboost import CatBoostRegressor, Pool
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential

# Loading the dataset
new_train_data = pd.read_csv('new_train_data.csv')
print(new_train_data.head())

# Splitting the dataset into features and target variable
x = new_train_data.iloc[:, :-1]
y = new_train_data.iloc[:, -1]

# Scaling the features
sc = MinMaxScaler()
x = sc.fit_transform(x)

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ----------------------------------
# Gradient Boosting Regressor
# ----------------------------------

# Initializing and training the Gradient Boosting Regressor model
Gradient_model = GradientBoostingRegressor()
Gradient_model.fit(x_train, y_train)

# Making predictions on the test set
y_pred_gb = Gradient_model.predict(x_test)

# Calculating evaluation metrics for Gradient Boosting Regressor
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mape_gb = mean_absolute_percentage_error(y_test, y_pred_gb)

# Printing the evaluation metrics for Gradient Boosting Regressor
print("Gradient Boosting Regressor Metrics:")
print(f"Mean Squared Error: {mse_gb}")
print(f"Root Mean Squared Error: {rmse_gb}")
print(f"R-squared Score: {r2_gb}")
print(f"Mean Absolute Percentage Error: {mape_gb}")

# Saving the trained Gradient Boosting Regressor model
joblib.dump(Gradient_model, 'Gradient_model.pkl')
print("Gradient Boosting Regressor Model trained and saved as 'Gradient_model.pkl'.")

# ----------------------------------
# LSTM (Long Short-Term Memory)
# ----------------------------------

# Reshaping data for LSTM
x_train_lstm = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test_lstm = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

# Defining the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, input_shape=(x_train_lstm.shape[1], x_train_lstm.shape[2])))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Training the LSTM model
model_lstm.fit(x_train_lstm, y_train, epochs=50, verbose=0)

# Making predictions with LSTM model
y_pred_lstm = model_lstm.predict(x_test_lstm)

# Reshape y_test for compatibility
y_test_lstm = y_test.values.reshape(-1, 1)

# Calculating evaluation metrics for LSTM
mse_lstm = mean_squared_error(y_test_lstm, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
r2_lstm = r2_score(y_test_lstm, y_pred_lstm)
mape_lstm = mean_absolute_percentage_error(y_test_lstm, y_pred_lstm)

# Printing the evaluation metrics for LSTM
print("\nLSTM Metrics:")
print(f"Mean Squared Error: {mse_lstm}")
print(f"Root Mean Squared Error: {rmse_lstm}")
print(f"R-squared Score: {r2_lstm}")
print(f"Mean Absolute Percentage Error: {mape_lstm}")

model_lstm.save('lstm_model.h5')
print("LSTM Model trained and saved as 'lstm_model.h5'.")

# ----------------------------------
# CatBoost Regressor
# ----------------------------------

# Initializing and training the CatBoost Regressor model
catboost_model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='RMSE')
catboost_model.fit(x_train, y_train, verbose=False)

# Making predictions on the test set
y_pred_catboost = catboost_model.predict(x_test)

# Calculating evaluation metrics for CatBoost Regressor
mse_catboost = mean_squared_error(y_test, y_pred_catboost)
rmse_catboost = np.sqrt(mse_catboost)
r2_catboost = r2_score(y_test, y_pred_catboost)
mape_catboost = mean_absolute_percentage_error(y_test, y_pred_catboost)

# Printing the evaluation metrics for CatBoost Regressor
print("\nCatBoost Regressor Metrics:")
print(f"Mean Squared Error: {mse_catboost}")
print(f"Root Mean Squared Error: {rmse_catboost}")
print(f"R-squared Score: {r2_catboost}")
print(f"Mean Absolute Percentage Error: {mape_catboost}")

# Saving the trained CatBoost Regressor model
catboost_model.save_model('catboost_model.cbm')
print("CatBoost Regressor Model trained and saved as 'catboost_model.cbm'.")
joblib.dump(sc, 'scaler.pkl')

# ----------------------------------
# Manual Prediction
# ----------------------------------

# Sample data for manual prediction (including all 10 features)
manual_data = np.array([[2, 642.15, 1403.14, 553.75, 47.49, 522.28, 8.4318, 392, 39.0, 23.4236]])
manual_data_scaled = sc.transform(manual_data)

# Gradient Boosting Regressor prediction
Gradient_model = joblib.load('Gradient_model.pkl')
manual_pred_gb = Gradient_model.predict(manual_data_scaled)
print(f"\nManual Prediction with Gradient Boosting Regressor: {manual_pred_gb[0]}")

# LSTM prediction
model_lstm = load_model('lstm_model.h5')
manual_data_lstm = manual_data_scaled.reshape((manual_data_scaled.shape[0], manual_data_scaled.shape[1], 1))
manual_pred_lstm = model_lstm.predict(manual_data_lstm)
print(f"Manual Prediction with LSTM: {manual_pred_lstm[0][0]}")

# CatBoost Regressor prediction
catboost_model = CatBoostRegressor()
catboost_model.load_model('catboost_model.cbm')
manual_pred_catboost = catboost_model.predict(manual_data_scaled)
print(f"Manual Prediction with CatBoost Regressor: {manual_pred_catboost[0]}")