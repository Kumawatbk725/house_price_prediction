# House Price Prediction

A machine learning project that predicts house prices based on various features. This project includes data preprocessing, model training, and deployment as a REST API.

## Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Preprocessed data files
├── models/               # Trained models and evaluation results
├── notebooks/            # Jupyter notebooks for data exploration and model development
├── src/                  # Source code
│   ├── preprocessing/    # Data preprocessing scripts
│   ├── training/         # Model training scripts
│   └── api/              # API-related code
├── tests/                # Test scripts
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── ProjectReport.md      # Project Report
└── app.py                # Flask application
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Processing

The project uses the California Housing dataset from scikit-learn. To process the data:

1. Run the data exploration notebook:
   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

2. Run the data preprocessing notebook:
   ```bash
   jupyter notebook notebooks/02_data_preprocessing.ipynb
   ```

## Model Training

To train and evaluate the machine learning models:

1. Run the model training notebook:
   ```bash
   jupyter notebook notebooks/03_model_training.ipynb
   ```

This will:
- Train several regression models (Linear Regression, Decision Tree, Random Forest, XGBoost)
- Perform hyperparameter tuning on the best model
- Evaluate model performance using RMSE, MAE, and R² scores
- Save the trained models to the `models/` directory

## API Deployment (continued)

### Running Locally

To run the API locally:

```bash
python app.py
```

The API will be available at http://localhost:5000.

### Docker Deployment

To build and run using Docker:

1. Build the Docker image:
   ```bash
   docker build -t house-price-prediction .
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 house-price-prediction
   ```

The API will be available at http://localhost:5000.

### Testing the API

You can test the API using the provided test script:

```bash
python tests/test_api.py
```

Alternatively, use curl:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984127, "AveBedrms": 1.02381, "Population": 322.0, "AveOccup": 2.555556, "Latitude": 37.88, "Longitude": -122.23}'
```

## Project Report

### Data Preprocessing

The preprocessing pipeline includes:
1. **Data Loading**: The California Housing dataset was loaded using scikit-learn
2. **Missing Value Handling**: Numerical missing values were filled with median values, and categorical missing values with the mode
3. **Feature Engineering**: Created derived features like rooms per household and population density
4. **Feature Scaling**: Applied StandardScaler to normalize all numerical features
5. **Data Splitting**: Split data into 80% training and 20% testing sets

### Model Selection and Optimization

Several regression models were trained and evaluated:
1. Linear Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The best performing model was then optimized using GridSearchCV to find the optimal hyperparameters. Evaluation metrics included:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

### Deployment Strategy

The trained model was deployed as a REST API using Flask, with the following features:
1. JSON input/output for easy integration
2. Error handling and logging
3. Health check endpoint
4. Docker containerization for easy deployment

## Future Improvements

Potential improvements to the project:
1. Implement more advanced feature engineering
2. Try more complex models like neural networks
3. Add cross-validation to improve model robustness
4. Implement CI/CD pipeline for automated testing and deployment
5. Add a simple web frontend for easier interaction

