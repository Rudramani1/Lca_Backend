from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from Lca_calculation import MetalLCA
from LCA import MiningDataPredictor

app = Flask(__name__)
CORS(app)

# Global variables for models and data
models = {}
data = None
feature_columns = []
target_columns = []
predictor = None

def load_and_train_models():
    """Load data and train models if not already done"""
    global models, data, feature_columns, target_columns, predictor
    
    if models:  # Already trained
        return
    
    try:
        # Initialize the predictor
        predictor = MiningDataPredictor('hello.csv')
        
        # Try to load saved models first
        if predictor.load_saved_models():
            print("✓ Successfully loaded pre-trained models")
            # Extract model info for compatibility
            models = {}
            for target_feature, model_info in predictor.trained_models.items():
                models[target_feature] = {
                    'model': model_info['model'],
                    'mae': 0,  # MAE not available from saved models
                    'trained_samples': 0
                }
            return
        else:
            print("No saved models found, training new models...")
            
            # Load and train new models
            if predictor.load_data():
                predictor.analyze_dataset()
                predictor.preprocess_data()
                predictor.train_predictive_models()  # This will also save models
                
                # Extract model info for compatibility
                models = {}
                for target_feature, model_info in predictor.trained_models.items():
                    models[target_feature] = {
                        'model': model_info['model'],
                        'mae': 0,  # MAE calculation would require separate evaluation
                        'trained_samples': 0
                    }
                
                print(f"Successfully trained {len(models)} models")
            else:
                raise Exception("Failed to load data for training")
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        raise e

def predict_missing_values(input_data):
    """Predict missing values using trained models"""
    predictions = {}
    
    if predictor is None:
        print("[ERROR] Predictor not initialized")
        return predictions
    
    try:
        # Use the predictor to predict all possible missing values
        predictions = predictor.predict_all_possible(input_data)
        print(f"Predicted missing values: {predictions}")
        
    except Exception as e:
        print(f"Error predicting missing values: {str(e)}")
        # Fallback to old method if needed
        for target, model_info in models.items():
            try:
                # Prepare input features
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                categorical_columns = ['Metal_Type', 'Mining_Method', 'Transport_Raw_Mode', 'Transport_EOL_Mode', 'Scrap_Type', 'Transport_Scrap_Mode']
                for col in categorical_columns:
                    if col in input_df.columns and input_df[col].dtype == 'object':
                        input_df[col] = pd.Categorical(input_df[col]).codes
                
                # Fill missing features with 0
                X = input_df[feature_columns].fillna(0)
                
                # Make prediction
                prediction = model_info['model'].predict(X)[0]
                predictions[target] = float(prediction)
                
            except Exception as e2:
                print(f"Error predicting {target}: {str(e2)}")
                predictions[target] = 0
    
    return predictions

@app.route('/api/predict', methods=['POST'])
def predict_and_calculate():
    """API endpoint to predict missing values and calculate LCA"""
    try:
        # Load and train models if not done
        load_and_train_models()
        
        # Get input data from request
        input_data = request.json
        
        # Predict missing values
        predictions = predict_missing_values(input_data)
        
        # Merge predictions with input data
        complete_data = {**input_data, **predictions}
        
        # Run LCA calculation
        lca_calculator = MetalLCA(**complete_data)
        lca_results = lca_calculator.run_lca()
        
        # Return both predictions and LCA results
        response = {
            'predictions': predictions,
            'lca_results': lca_results,
            'complete_data': complete_data,
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'models_trained': len(models)})

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get the list of features that the model was trained on"""
    try:
        if predictor is None:
            load_and_train_models()
        
        if predictor is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
        # Get all features that the models were trained on
        all_features = set()
        
        # Add categorical features
        all_features.update(predictor.categorical_features)
        
        # Add numerical features (excluding target columns that are predicted)
        target_columns = ['Collection_Rate_%', 'Recycling_Rate_%', 'Landfilling_Rate_%', 'Recovery_Efficiency_%']
        input_features = [f for f in predictor.numerical_features if f not in target_columns]
        all_features.update(input_features)
        
        # Get feature statistics for ranges
        feature_info = {}
        for feature in all_features:
            if feature in predictor.feature_stats:
                feature_info[feature] = {
                    'min': float(predictor.feature_stats[feature]['min']),
                    'max': float(predictor.feature_stats[feature]['max']),
                    'type': 'categorical' if feature in predictor.categorical_features else 'numerical'
                }
        
        return jsonify({
            'features': list(all_features),
            'feature_info': feature_info,
            'target_columns': target_columns,
            'categorical_features': predictor.categorical_features,
            'numerical_features': input_features
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    # Load and train models on startup
    print("Loading and training models...")
    load_and_train_models()
    print("Starting Flask server...")
    app.run(debug=True, port=5000)