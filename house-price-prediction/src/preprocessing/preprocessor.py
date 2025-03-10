import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load raw data from CSV file"""
        return pd.read_csv(self.input_file)
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # For numerical columns, fill with median
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def feature_engineering(self, df):
        """Create new features based on domain knowledge"""
        # Example: Calculate rooms per household
        if 'AveRooms' in df.columns and 'HouseHold' in df.columns:
            df['RoomsPerHousehold'] = df['AveRooms'] / df['HouseHold']
        
        # Example: Calculate population density
        if 'Population' in df.columns and 'AveOccup' in df.columns:
            df['PopulationDensity'] = df['Population'] / df['AveOccup']
        
        return df
    
    def scale_features(self, df_train, df_test):
        """Scale numerical features using StandardScaler"""
        # Identify numerical columns (excluding the target variable)
        num_cols = df_train.select_dtypes(include=np.number).columns.tolist()
        if 'Price' in num_cols:
            num_cols.remove('Price')
        
        # Fit scaler on training data
        self.scaler.fit(df_train[num_cols])
        
        # Transform both training and testing data
        df_train_scaled = df_train.copy()
        df_test_scaled = df_test.copy()
        
        df_train_scaled[num_cols] = self.scaler.transform(df_train[num_cols])
        df_test_scaled[num_cols] = self.scaler.transform(df_test[num_cols])
        
        # Save the scaler for later use in prediction
        joblib.dump(self.scaler, os.path.join(self.output_dir, 'scaler.pkl'))
        
        return df_train_scaled, df_test_scaled
    
    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X = df.drop('Price', axis=1)
        y = df['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Reconstruct DataFrames
        df_train = pd.concat([X_train, pd.DataFrame(y_train, index=X_train.index, columns=['Price'])], axis=1)
        df_test = pd.concat([X_test, pd.DataFrame(y_test, index=X_test.index, columns=['Price'])], axis=1)
        
        return df_train, df_test
    
    def preprocess(self):
        """Execute the entire preprocessing pipeline"""
        # Load data
        df = self.load_data()
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Split data
        df_train, df_test = self.split_data(df)
        
        # Scale features
        df_train_scaled, df_test_scaled = self.scale_features(df_train, df_test)
        
        # Save processed datasets
        df_train_scaled.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
        df_test_scaled.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)
        
        print(f"Preprocessing complete. Files saved to {self.output_dir}")
        return df_train_scaled, df_test_scaled

if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        input_file='../../data/raw/california_housing.csv',
        output_dir='../../data/processed'
    )
    train_data, test_data = preprocessor.preprocess()