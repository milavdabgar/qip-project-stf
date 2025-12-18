#!/usr/bin/env python3
"""
Prediction script for System Threat Forecaster
Loads trained models and generates predictions for web app
"""

import sys
import json
import joblib
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Paths to saved models
ML_MODELS_PATH = 'saved_models/ml_models.pkl'
DL_MODELS_PATH = 'saved_models/dl_models.pth'
PREPROCESSORS_PATH = 'saved_models/preprocessors.pkl'

def load_models():
    """Load all trained models and preprocessors."""
    models = {
        'ml_models': None,
        'dl_models': None,
        'preprocessors': None,
        'device': None
    }
    
    try:
        # Load ML models (scikit-learn models saved with joblib)
        try:
            models['ml_models'] = joblib.load(ML_MODELS_PATH)
            print(f"Loaded {len(models['ml_models'])} ML models", file=sys.stderr)
        except FileNotFoundError:
            print(f"ML models not found at {ML_MODELS_PATH}", file=sys.stderr)
        
        # Load preprocessors
        try:
            models['preprocessors'] = joblib.load(PREPROCESSORS_PATH)
            print("Loaded preprocessors", file=sys.stderr)
        except FileNotFoundError:
            print(f"Preprocessors not found at {PREPROCESSORS_PATH}", file=sys.stderr)
        
        # Load DL models (PyTorch models)
        try:
            # Determine device
            if torch.backends.mps.is_available():
                models['device'] = torch.device('mps')
            elif torch.cuda.is_available():
                models['device'] = torch.device('cuda')
            else:
                models['device'] = torch.device('cpu')
            
            checkpoint = torch.load(DL_MODELS_PATH, map_location=models['device'])
            models['dl_models'] = checkpoint['models']
            print(f"Loaded {len(models['dl_models'])} DL models on {models['device']}", file=sys.stderr)
        except FileNotFoundError:
            print(f"DL models not found at {DL_MODELS_PATH}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading DL models: {e}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        sys.exit(1)
    
    return models

def preprocess_features(features, preprocessors):
    """Preprocess raw features using saved preprocessors."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Apply label encoding to categorical features
        if 'label_encoders' in preprocessors:
            for col, encoder in preprocessors['label_encoders'].items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except:
                        # Unknown category, use -1
                        df[col] = -1
        
        # Apply scaling
        if 'scaler' in preprocessors:
            X = preprocessors['scaler'].transform(df)
        else:
            X = df.values
        
        return X
    
    except Exception as e:
        print(f"Error preprocessing features: {e}", file=sys.stderr)
        raise

def predict_ml(models_dict, X):
    """Get predictions from ML models."""
    predictions = {}
    
    for name, model in models_dict.items():
        try:
            # Get prediction
            pred = int(model.predict(X)[0])
            
            # Get confidence (probability)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = float(proba[pred])
            else:
                # Some models don't have probabilities
                confidence = 1.0
            
            predictions[name] = {
                'prediction': pred,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Error predicting with {name}: {e}", file=sys.stderr)
    
    return predictions

def predict_dl(models_dict, X, device):
    """Get predictions from DL models."""
    predictions = {}
    
    try:
        X_tensor = torch.FloatTensor(X).to(device)
        
        for name, model_dict in models_dict.items():
            try:
                model = model_dict['model']
                model.eval()
                
                with torch.no_grad():
                    output = model(X_tensor)
                    proba = torch.softmax(output, dim=1)
                    pred = torch.argmax(proba, dim=1).item()
                    confidence = float(proba[0, pred])
                
                predictions[name] = {
                    'prediction': int(pred),
                    'confidence': confidence
                }
            except Exception as e:
                print(f"Error predicting with {name}: {e}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error in DL prediction: {e}", file=sys.stderr)
    
    return predictions

def main():
    """Main prediction function."""
    try:
        # Read input from stdin
        input_data = sys.stdin.read()
        request = json.loads(input_data)
        features = request['features']
        
        # Load models
        models = load_models()
        
        # Preprocess features
        X = preprocess_features(features, models['preprocessors'])
        
        # Get predictions
        ml_predictions = {}
        dl_predictions = {}
        
        if models['ml_models']:
            ml_predictions = predict_ml(models['ml_models'], X)
        
        if models['dl_models'] and models['device']:
            dl_predictions = predict_dl(models['dl_models'], X, models['device'])
        
        # Return results as JSON
        result = {
            'ml_models': ml_predictions,
            'dl_models': dl_predictions
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'ml_models': {},
            'dl_models': {}
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()
