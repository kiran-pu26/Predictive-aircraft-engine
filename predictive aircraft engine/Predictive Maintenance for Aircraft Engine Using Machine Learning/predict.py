# predict.py

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load('rul_model.pkl')

# Load and preprocess new data
data_new = pd.read_csv('CMaps/new_data.txt', sep=" ")  # Update the path as needed
data_new.drop(columns=['Unnamed: 26', 'Unnamed: 27'], inplace=True)
col_names = [
    'engine', 'cycle', 'setting_1', 'setting_2', 'setting_3',
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20',
    'sensor_21'
]
data_new.columns = col_names

# Scale the new data
scaler = MinMaxScaler()
data_new_scaled = pd.DataFrame(scaler.fit_transform(data_new), columns=col_names)

# Predict
X_new = data_new_scaled.drop(columns=['cycle'])
y_new_pred = model.predict(X_new)

# Print predictions
print("Predicted RUL for new data:")
print(y_new_pred)
