import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import joblib # Used for saving/loading the model

warnings.filterwarnings('ignore')

class MiningDataPredictor:
    """
    A class to load, process, train, and use predictive models for mining data.
    """
    def __init__(self, csv_file='hello.csv'):
        self.csv_file = csv_file
        self.data = None
        self.processed_data = None
        self.label_encoders = {}
        self.trained_models = {}
        self.feature_stats = {}
        # The full list of expected categorical columns
        self.categorical_features = [
            'Metal_Type', 'Mining_Method', 'Transport_Raw_Mode', 
            'Scrap_Type', 'Transport_Scrap_Mode'
        ]
        self.numerical_features = []
    
    def load_saved_models(self):
        """Load trained models and metadata from disk"""
        print("[INFO] Loading saved models from disk...")
        try:
            models_dir = Path('models')
            
            if not models_dir.exists():
                print("[ERROR] Models directory not found!")
                return False
            
            # Load label encoders
            encoders_path = models_dir / "label_encoders.joblib"
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
                print(f"       Loaded label encoders from {encoders_path}")
            
            # Load feature statistics
            stats_path = models_dir / "feature_stats.joblib"
            if stats_path.exists():
                self.feature_stats = joblib.load(stats_path)
                print(f"       Loaded feature statistics from {stats_path}")
            
            # Load features info
            features_path = models_dir / "features_info.joblib"
            if features_path.exists():
                features_info = joblib.load(features_path)
                self.categorical_features = features_info['categorical_features']
                self.numerical_features = features_info['numerical_features']
                print(f"       Loaded features info from {features_path}")
            
            # Load trained models
            self.trained_models = {}
            for model_file in models_dir.glob("*_model.joblib"):
                if model_file.name != "label_encoders.joblib" and not model_file.name.startswith("feature_"):
                    target_feature = model_file.stem.replace("_model", "")
                    try:
                        model_info = joblib.load(model_file)
                        self.trained_models[target_feature] = model_info
                        print(f"       Loaded model for '{target_feature}' from {model_file}")
                    except Exception as e:
                        print(f"[WARNING] Failed to load model {model_file}: {str(e)}")
            
            print(f"✓ Successfully loaded {len(self.trained_models)} models!")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {str(e)}")
            return False

    def load_data(self):
        print("[INFO] Loading and inspecting data...")
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"       Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            if 'Record_ID' in self.data.columns:
                self.data = self.data.drop('Record_ID', axis=1)
            
            self.numerical_features = self.data.select_dtypes(include=np.number).columns.tolist()
            
            # --- START OF DIAGNOSTIC CHECK ---
            
            # This is the full list we expect to find
            expected_categoricals = [
                'Metal_Type', 'Mining_Method', 'Transport_Raw_Mode', 
                'Scrap_Type', 'Transport_Scrap_Mode'
            ]
            
            # Filter the list to only those that exist in the CSV
            found_categoricals = [col for col in expected_categoricals if col in self.data.columns]
            missing_categoricals = list(set(expected_categoricals) - set(found_categoricals))

            print("\n[DIAGNOSTIC CHECK] Verifying categorical columns...")
            print(f"       > Script expects: {expected_categoricals}")
            print(f"       > Columns found in '{self.csv_file}': {found_categoricals}")
            
            if missing_categoricals:
                print(f"       > ❌ Columns NOT FOUND: {missing_categoricals}")
                print("       > Please check for typos in your CSV headers or make sure you are using the correct file.")
            else:
                print("       > ✅ All categorical columns were found successfully!")
            
            self.categorical_features = found_categoricals # Set the final list to only what was found
            print("-" * 50) # Separator to make the log clearer
            
            # --- END OF DIAGNOSTIC CHECK ---

            if self.data.isnull().sum().sum() == 0:
                print("       ✓ Dataset is complete - no missing values found!")
            else:
                print("       ! Warning: Missing values detected.")
            return True
        except FileNotFoundError:
            print(f"[ERROR] File '{self.csv_file}' not found!")
            print(f"       > The script is looking for a file named '{self.csv_file}' in the same directory.")
            return False

    def analyze_dataset(self):
        print("\n[INFO] Analyzing dataset and generating correlation heatmap...")
        key_features = [
            'Quantity_tons', 'Ore_Grade_%', 'Energy_Mining_MJ', 'Energy_Recycling_MJ',
            'Waste_Generated_tons', 'Electricity_kWh', 'Emissions_CO_kg', 'Material_Yield_%',
            'Recycling_Yield_%', 'Landfilling_Rate_%'
        ]
        key_features_exist = [f for f in key_features if f in self.data.columns]
        
        correlation_matrix = self.data[key_features_exist].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Key Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        print("       ✓ Key feature correlation heatmap saved to 'correlation_heatmap.png'")

    def preprocess_data(self):
        print("\n[INFO] Preprocessing data (encoding categorical features)...")
        self.processed_data = self.data.copy()
        for col in self.categorical_features:
            le = LabelEncoder()
            self.processed_data[f'{col}_encoded'] = le.fit_transform(self.processed_data[col])
            self.label_encoders[col] = le
            print(f"       Encoded '{col}' with {len(le.classes_)} unique values.")
        
        for feature in self.numerical_features:
            stats = self.processed_data[feature]
            self.feature_stats[feature] = {'min': stats.min(), 'max': stats.max()}

    def train_predictive_models(self):
        print(f"\n[INFO] Training {len(self.numerical_features)} models on the training data...")
        for i, target_feature in enumerate(self.numerical_features):
            self._train_single_feature_model(target_feature)
        print(f"\n[SUCCESS] Successfully trained {len(self.trained_models)} models.")
        
        # Save all models to disk
        self.save_models()
    
    def save_models(self):
        """Save trained models and encoders to disk"""
        print("\n[INFO] Saving models to disk...")
        try:
            # Create models directory if it doesn't exist
            models_dir = Path('models')
            models_dir.mkdir(exist_ok=True)
            
            # Save each trained model
            for target_feature, model_info in self.trained_models.items():
                model_path = models_dir / f"{target_feature}_model.joblib"
                joblib.dump(model_info, model_path)
                print(f"       Saved model for '{target_feature}' to {model_path}")
            
            # Save label encoders
            encoders_path = models_dir / "label_encoders.joblib"
            joblib.dump(self.label_encoders, encoders_path)
            print(f"       Saved label encoders to {encoders_path}")
            
            # Save feature statistics
            stats_path = models_dir / "feature_stats.joblib"
            joblib.dump(self.feature_stats, stats_path)
            print(f"       Saved feature statistics to {stats_path}")
            
            # Save categorical and numerical features lists
            features_info = {
                'categorical_features': self.categorical_features,
                'numerical_features': self.numerical_features
            }
            features_path = models_dir / "features_info.joblib"
            joblib.dump(features_info, features_path)
            print(f"       Saved features info to {features_path}")
            
            print("✓ All models and metadata saved successfully!")
            
        except Exception as e:
            print(f"[ERROR] Failed to save models: {str(e)}")

    def _train_single_feature_model(self, target_feature):
        all_predictors = [f for f in self.numerical_features if f != target_feature]
        all_predictors += [f'{c}_encoded' for c in self.categorical_features]
        
        # Use the combined processed data (numerical + encoded categorical) for correlation
        temp_df_for_corr = self.processed_data[all_predictors + [target_feature]]
        correlations = temp_df_for_corr.corr(numeric_only=True)[target_feature].abs()

        # Drop the target itself from the list of potential predictors
        correlations = correlations.drop(target_feature, errors='ignore')

        top_predictors = correlations.nlargest(10).index.tolist()
        
        X = self.processed_data[top_predictors]
        y = self.processed_data[target_feature]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=2)
        model.fit(X_train, y_train)
        
        self.trained_models[target_feature] = {
            'model': model, 
            'predictor_features': top_predictors
        }

    def predict_all_possible(self, input_data):
        known_data = input_data.copy()
        predictions = {}
        
        sorted_models = sorted(self.trained_models.items(), key=lambda item: len(item[1]['predictor_features']))

        while True:
            made_new_prediction = False
            for target_feature, model_info in sorted_models:
                if target_feature in known_data:
                    continue
                
                required = model_info['predictor_features']
                can_predict = all(feat.replace('_encoded', '') in known_data for feat in required)
                
                if can_predict:
                    input_df = pd.DataFrame([known_data])
                    for col, le in self.label_encoders.items():
                        if col in input_df.columns:
                            if input_df[col].iloc[0] not in le.classes_:
                                most_frequent_class = self.data[col].mode()[0]
                                transformed_val = le.transform([most_frequent_class])[0]
                                input_df[f'{col}_encoded'] = transformed_val
                            else:
                                input_df[f'{col}_encoded'] = le.transform(input_df[col])
                    
                    prediction = model_info['model'].predict(input_df[required])[0]
                    stats = self.feature_stats[target_feature]
                    clipped_prediction = np.clip(prediction, stats['min'], stats['max'])
                    known_data[target_feature] = clipped_prediction
                    predictions[target_feature] = clipped_prediction
                    made_new_prediction = True

            if not made_new_prediction:
                break
        return predictions

def run_evaluation(predictor, test_df):
    """
    Evaluates the model's performance on an unseen test set.
    """
    print("\n" + "="*60)
    print("      EVALUATING MODEL ON UNSEEN TEST DATA")
    print("="*60)

    if len(test_df) > 3:
        test_samples = test_df.sample(3, random_state=42)
    else:
        test_samples = test_df

    target_to_predict = 'Energy_Consumption_MJ'

    for i, (_, test_row) in enumerate(test_samples.iterrows()):
        print(f"\n--- Running Test Example {i+1} ---")
        
        actual_value = test_row[target_to_predict]
        input_data = test_row.drop(target_to_predict).to_dict()

        print("INPUT DATA (from unseen test set):")
        print(f"  - Metal_Type: {input_data.get('Metal_Type')}")
        print(f"  - Primary Production Energy (Mining_MJ): {input_data.get('Energy_Mining_MJ'):,.0f}")
        print(f"  - Secondary Production Energy (Recycling_MJ): {input_data.get('Energy_Recycling_MJ'):,.0f}")
        print("  - ... and other features")

        predictions = predictor.predict_all_possible(input_data)
        
        print("\nEVALUATION RESULTS:")
        predicted_value = predictions.get(target_to_predict)

        if predicted_value is not None:
            error = abs(predicted_value - actual_value)
            print(f"  - ✅ Predicted '{target_to_predict}': {predicted_value:,.2f}")
            print(f"  - 🎯 Actual Value: {actual_value:,.2f}")
            print(f"  - 📏 Absolute Error: {error:,.2f}")
        else:
            print(f"  - ❌ Could not predict '{target_to_predict}' with the provided data.")

def main():
    # The script now looks for 'hello.csv' by default
    DATA_FILE = 'hello.csv'

    try:
        full_data = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"[ERROR] The data file '{DATA_FILE}' was not found. Please make sure it's in the same directory as the script.")
        return
        
    if 'Record_ID' in full_data.columns:
        full_data = full_data.drop('Record_ID', axis=1)

    train_df, test_df = train_test_split(full_data, test_size=0.2, random_state=42)
    print(f"[INFO] Data split into {len(train_df)} training rows and {len(test_df)} testing rows.")

    # Initialize predictor with the correct file name
    predictor = MiningDataPredictor(csv_file=DATA_FILE) 
    
    # Manually assign the training data for processing and training
    predictor.data = train_df 
    predictor.numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
    # The categorical features will be filtered inside the .preprocess_data() call after it re-loads the data
    # Let's adjust this to be more direct
    predictor.categorical_features = [col for col in predictor.categorical_features if col in train_df.columns]


    predictor.analyze_dataset()
    predictor.preprocess_data()
    predictor.train_predictive_models()
    
    run_evaluation(predictor, test_df)
    
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()