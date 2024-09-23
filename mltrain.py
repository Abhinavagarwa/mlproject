import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('/Users/abhinavagarwal/Documents/air/archive/city_day.csv')

# Convert 'date' to datetime and sort data by date
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date', inplace=True)

# Encode the 'city' column
label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])

# Drop irrelevant columns and handle missing values
data.fillna(method='ffill', inplace=True)

# Feature selection (all relevant pollutants and city as features)
features = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'AQI'

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])

# Prepare the data for time series forecasting (e.g., using the last 7 days to predict the AQI)
def create_sequences(features, target, time_steps=7):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i+time_steps])
        y.append(target[i+time_steps])
    return np.array(X), np.array(y)

# Define the time step for the LSTM (e.g., using the past 7 days of data)
time_steps = 7

# Prepare sequences
X, y = create_sequences(scaled_features, data[target].values, time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape input data to be 3D [samples, time steps, features] for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions
y_pred = model.predict(X_test)

# Plotting the results
plt.figure(figsize=(10,6))
plt.plot(y_test, label='True AQI')
plt.plot(y_pred, label='Predicted AQI')
plt.title('AQI Prediction using LSTM')
plt.xlabel('Time')
plt.ylabel('AQI')
plt.legend()
plt.show()

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print(f"Test MSE: {mse}")
model.save('model.h5')