import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, train_file, test_file, models_dir):
        self.train_file = train_file
        self.test_file = test_file
        self.models_dir = models_dir
        
        # Initialize empty variables
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
    def load_data(self):
        """Load preprocessed training and testing data"""
        train_df = pd.read_csv(self.train_file)
        test_df = pd.read_csv(self.test_file)
        
        self.X_train = train_df.drop('Price', axis=1)
        self.y_train = train_df['Price']
        
        self.X_test = test_df.drop('Price', axis=1)
        self.y_test = test_df['Price']
        
        print(f"Loaded training data: {self.X_train.shape}, {self.y_train.shape}")
        print(f"Loaded testing data: {self.X_test.shape}, {self.y_test.shape}")
        
    def train_models(self):
        """Train multiple regression models"""
        models = {
            'linear_regression': LinearRegression(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(random_state=42),
            'xgboost': XGBRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
        
        # Identify best model based on RMSE
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\nBest model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.4f}")
        
        return results, best_model[0]
    
    def hyperparameter_tuning(self, model_name):
        """Perform hyperparameter tuning on the selected model"""
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'linear_regression':
            # Linear Regression has few hyperparameters to tune
            model = LinearRegression()
            param_grid = {
                'fit_intercept': [True, False],
                'normalize': [True, False] if hasattr(LinearRegression(), 'normalize') else {}
            }
        
        elif model_name == 'decision_tree':
            model = DecisionTreeRegressor(random_state=42)
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        
        elif model_name == 'random_forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        elif model_name == 'xgboost':
            model = XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Run grid search with cross-validation
        grid_search = GridSearchCV(
            model, param_grid, cv=5, 
            scoring='neg_mean_squared_error', verbose=1, n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        
        # Evaluate the tuned model
        y_pred = best_model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Tuned {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Save the tuned model
        model_path = os.path.join(self.models_dir, f"{model_name}_tuned.pkl")
        joblib.dump(best_model, model_path)
        print(f"Tuned model saved to {model_path}")
        
        return best_model, {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': grid_search.best_params_
        }
    
    def evaluate_model(self, model, plot=True):
        """Evaluate model and generate plots"""
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Final Model - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        if plot:
            # Actual vs Predicted plot
            plt.figure(figsize=(10, 6))
            plt.scatter(self.y_test, y_pred, alpha=0.5)
            plt.plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--')
            plt.xlabel('Actual Prices')
            plt.ylabel('Predicted Prices')
            plt.title('Actual vs Predicted House Prices')
            
            # Save plot
            plt.savefig(os.path.join(self.models_dir, 'actual_vs_predicted.png'))
            
            # Residuals plot
            plt.figure(figsize=(10, 6))
            residuals = self.y_test - y_pred
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Prices')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            
            # Save plot
            plt.savefig(os.path.join(self.models_dir, 'residuals.png'))
            
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def feature_importance(self, model):
        """Extract and visualize feature importance if available"""
        feature_names = self.X_train.columns
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importances')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Relative Importance')
            
            # Save plot
            plt.savefig(os.path.join(self.models_dir, 'feature_importance.png'))
            
            # Print feature importance
            print("\nFeature Importance:")
            for i, idx in enumerate(indices):
                print(f"{i+1}. {feature_names[idx]} - {importances[idx]:.4f}")
            
            return dict(zip(feature_names, importances))
        else:
            print("Feature importance not available for this model")
            return None
    
    def train_and_optimize(self):
        """Execute the entire model training pipeline"""
        # Load data
        self.load_data()
        
        # Train initial models
        results, best_model_name = self.train_models()
        
        # Hyperparameter tuning on the best model
        best_model, tuning_results = self.hyperparameter_tuning(best_model_name)
        
        # Evaluate the final model
        evaluation = self.evaluate_model(best_model)
        
        # Feature importance
        importance = self.feature_importance(best_model)
        
        return best_model, {
            'model_name': best_model_name,
            'evaluation': evaluation,
            'tuning_results': tuning_results,
            'feature_importance': importance
        }

if __name__ == "__main__":
    trainer = ModelTrainer(
        train_file='../../data/processed/train.csv',
        test_file='../../data/processed/test.csv',
        models_dir='../../models'
    )
    
    best_model, results = trainer.train_and_optimize()