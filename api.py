from flask import Flask,jsonify,request,render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import tensorflow as tf

app=Flask(__name__)
CORS(app)

model=tf.keras.models.load_model('./model_final.h5')
print("model fetched")
test_data = np.array([[75, 120.51, 40, 35, 50, 30, 0.7, 20, 3.93]])
test_prediction = model.predict(test_data)
print("Test Prediction (AQI):", test_prediction.flatten())

feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'Toluene']

@app.route("/predict",methods=['POST','GET'])
def members():
    try:
        data = request.get_json()
        print("Received data:", data)
        user_input = pd.DataFrame(data, index=[0])  
        user_input = user_input[feature_columns]

        predictions = model.predict(user_input)
        print("Processed input for prediction:", user_input)
        return jsonify({"predicted_aqi": predictions.flatten().tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__=="__main__":
    app.run(debug=True)