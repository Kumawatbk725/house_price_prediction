import requests
import json
import time
import os

def test_api_locally():
    """Test the API running locally"""
    url = 'http://localhost:5000/predict'
    
    # Test data
    test_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    # Convert to JSON
    json_data = json.dumps(test_data)
    
    # Make POST request
    try:
        response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})
        
        # Check status code
        print(f"Status Code: {response.status_code}")
        
        # Print response
        if response.status_code == 200:
            result = response.json()
            print(f"Predicted Price: {result['predicted_price']}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure it's running.")

def test_api_with_curl():
    """Generate curl command for testing the API"""
    test_data = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    
    # Convert to JSON string with proper escaping for command line
    json_str = json.dumps(test_data).replace('"', '\\"')
    
    # Generate curl command
    curl_cmd = f'curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{json_str}"'
    
    print("Run the following curl command to test the API:")
    print(curl_cmd)

if __name__ == "__main__":
    # Test with Python requests
    test_api_locally()
    
    # Print curl command for manual testing
    print("\n")
    test_api_with_curl()