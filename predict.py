#!/usr/bin/env python3
"""
Prediction script for System Threat Forecaster
Loads trained models and generates predictions for web app
"""

import sys
import json
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Define PyTorch model architectures (must match training)
class DeepMLP(nn.Module):
    """Deep MLP with Batch Normalization."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3, use_batch_norm=True):
        super(DeepMLP, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, dim, dropout_rate=0.3, use_batch_norm=True):
        super(ResidualBlock, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = x
        
        out = self.fc1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        if self.use_batch_norm:
            out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out

class ResidualNet(nn.Module):
    """Neural Network with Residual Connections."""
    
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.3, use_batch_norm=True):
        super(ResidualNet, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim) if use_batch_norm else None
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate, use_batch_norm)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.input_layer(x)
        if self.bn_input is not None:
            x = self.bn_input(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x

class AttentionBlock(nn.Module):
    """Self-attention block for feature interactions."""
    
    def __init__(self, dim, num_heads=4, dropout_rate=0.3):
        super(AttentionBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        x = x.squeeze(1)
        return x

class AttentionNet(nn.Module):
    """Neural Network with Attention Mechanism."""
    
    def __init__(self, input_dim, hidden_dim=256, num_blocks=2, num_heads=4, dropout_rate=0.3):
        super(AttentionNet, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        for block in self.attention_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x

class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(self, input_dim, d_token=192, n_blocks=3, attention_heads=8, 
                 ffn_factor=4, dropout_rate=0.1):
        super(FTTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_token = d_token
        
        self.feature_tokenizer = nn.Linear(1, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        self.positional_embeddings = nn.Parameter(torch.randn(1, input_dim + 1, d_token))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_heads,
            dim_feedforward=d_token * ffn_factor,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        self.output_layer = nn.Linear(d_token, 2)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.unsqueeze(-1)
        x = self.feature_tokenizer(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.positional_embeddings
        
        x = self.transformer_encoder(x)
        
        cls_output = x[:, 0, :]
        
        output = self.output_layer(cls_output)
        return output

# Get the script's directory and project root
SCRIPT_DIR = Path(__file__).parent.absolute()

# Paths to saved models (relative to script location)
ML_MODELS_PATH = SCRIPT_DIR / 'saved_models' / 'ml_models.pkl'
DL_MODELS_PATH = SCRIPT_DIR / 'saved_models' / 'dl_models.pth'
PREPROCESSORS_PATH = SCRIPT_DIR / 'saved_models' / 'preprocessors.pkl'

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
            
            checkpoint = torch.load(DL_MODELS_PATH, map_location=models['device'], weights_only=False)
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
        
        print(f"Input features: {df.shape[1]} columns", file=sys.stderr)
        print(f"Input columns (first 10): {list(df.columns)[:10]}", file=sys.stderr)
        
        # Reorder columns to match training data
        if 'feature_columns' in preprocessors:
            expected_cols = preprocessors['feature_columns']
            print(f"Expected features: {len(expected_cols)} columns", file=sys.stderr)
            print(f"Expected columns (first 10): {expected_cols[:10]}", file=sys.stderr)
            
            # Add missing columns with default value
            for col in expected_cols:
                if col not in df.columns:
                    print(f"Missing column: {col}", file=sys.stderr)
                    df[col] = 0
            
            # Remove extra columns
            extra_cols = [col for col in df.columns if col not in expected_cols]
            if extra_cols:
                print(f"Extra columns: {extra_cols[:5]}", file=sys.stderr)
                df = df.drop(columns=extra_cols)
            
            # Reorder to match training
            df = df[expected_cols]
            print(f"After reordering: {df.shape[1]} columns", file=sys.stderr)
        
        # Apply label encoding to categorical features FIRST (before scaling)
        if 'label_encoders' in preprocessors:
            for col, encoder in preprocessors['label_encoders'].items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except Exception as e:
                        # Unknown category, use -1
                        print(f"Unknown category in {col}, using -1", file=sys.stderr)
                        df[col] = -1
        
        # Apply scaling - only to numeric columns (after label encoding)
        if 'scaler' in preprocessors:
            # Get numeric columns (after label encoding, all are numeric)
            if 'numeric_columns' in preprocessors:
                numeric_cols = preprocessors['numeric_columns']
                print(f"Scaling {len(numeric_cols)} numeric columns", file=sys.stderr)
                # Scale only numeric columns
                df[numeric_cols] = preprocessors['scaler'].transform(df[numeric_cols].values)
            else:
                # Fallback: scale all columns (old format)
                df = pd.DataFrame(
                    preprocessors['scaler'].transform(df.values),
                    columns=df.columns,
                    index=df.index
                )
        
        X = df.values
        return X
    
    except Exception as e:
        print(f"Error preprocessing features: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
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
            print(f"Running ML predictions with {len(models['ml_models'])} models", file=sys.stderr)
            ml_predictions = predict_ml(models['ml_models'], X)
            print(f"ML predictions: {list(ml_predictions.keys())}", file=sys.stderr)
        else:
            print("No ML models loaded", file=sys.stderr)
        
        if models['dl_models'] and models['device']:
            print(f"Running DL predictions with {len(models['dl_models'])} models", file=sys.stderr)
            dl_predictions = predict_dl(models['dl_models'], X, models['device'])
            print(f"DL predictions: {list(dl_predictions.keys())}", file=sys.stderr)
        else:
            print("No DL models loaded", file=sys.stderr)
        
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
