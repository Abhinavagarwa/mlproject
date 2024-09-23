from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)
# Load the saved LSTM model
model = load_model('/Users/abhinavagarwal/Documents/air/model.h5')

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
# Load the dataset (should include past pollution data)
data = pd.read_csv('/Users/abhinavagarwal/Documents/air/archive/city_day.csv')

# Initialize the Flask app
app = Flask(__name__)

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict_aqi():
    # Get the input data from the request (city and date)
    city = request.json['City']
    date = request.json['Date']

    # Convert the input date to a datetime object
    date = pd.to_datetime(date)

    # Encode the city using the same LabelEncoder used during training
    label_encoder = LabelEncoder()
    data['City'] = label_encoder.fit_transform(data['City'])

    # Find the encoded city value
    encoded_city = label_encoder.transform([city])[0]

    # Extract the past 7 days of data for the given city and date
    date_range = pd.date_range(end=date, periods=7)

    # Filter the dataset for the past 7 days and the given city
    city_data = data[(data['City'] == encoded_city) & (data['Date'].isin(date_range))]

    if city_data.shape[0] < 7:
        return jsonify({'error': 'Not enough historical data to make a prediction.'})

    # Select the relevant features for the model
    features = ['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
    historical_features = city_data[features].values

    # Scale the features using the same scaler used during training
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(historical_features)

    # Reshape the data to the format expected by the LSTM (1 sample, 7 time steps, number of features)
    scaled_features = np.reshape(scaled_features, (1, scaled_features.shape[0], scaled_features.shape[1]))

    # Make the prediction using the LSTM model
    predicted_aqi = model.predict(scaled_features)

    # Return the prediction as a JSON response
    return jsonify({'predicted_AQI': float(predicted_aqi[0][0])})

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5004)
