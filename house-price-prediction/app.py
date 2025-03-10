import os
import json
import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Global variables
MODEL_PATH = os.path.join('models', 'xgboost_tuned.pkl')  # Update with your best model
SCALER_PATH = os.path.join('data', 'processed', 'scaler.pkl')

# Global variables for model and scaler
model = None
scaler = None
feature_names = None

# Load model and scaler function
def load_model_and_scaler():
    global model, scaler, feature_names
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        logger.info(f"Loading scaler from {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        
        # Get feature names from sample data
        sample_data = pd.read_csv(os.path.join('data', 'processed', 'train.csv'))
        feature_names = sample_data.drop('Price', axis=1).columns.tolist()
        
        logger.info(f"Model and scaler loaded successfully. Features: {feature_names}")
    except Exception as e:
        logger.error(f"Error loading model or scaler: {str(e)}")
        raise

# Load model when app is initialized
with app.app_context():
    load_model_and_scaler()

@app.route('/')
def home():
    return """
    <h1>House Price Prediction API</h1>
    <p>Send a POST request to /predict with JSON data containing house features.</p>
    <p>Example:</p>
    <pre>
    {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json(force=True)
        logger.info(f"Received prediction request with data: {data}")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Check for missing features and add them with defaults if necessary
        for feature in feature_names:
            if feature not in input_df.columns:
                logger.warning(f"Feature {feature} missing from input. Using default value 0.")
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Apply scaling
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Log and return result
        result = {'predicted_price': float(prediction)}
        logger.info(f"Prediction result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        error_message = f"Error making prediction: {str(e)}"
        logger.error(error_message)
        return jsonify({'error': error_message}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)