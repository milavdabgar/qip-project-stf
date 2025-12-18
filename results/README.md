# Model Performance Results

This directory contains comprehensive performance metrics and training results for all models in the System Threat Forecaster project.

## Files

### model_performance.json
**Centralized model performance database** - Contains complete training metrics, hyperparameters, and configurations for all ML and DL models.

#### Structure:
```json
{
  "metadata": {
    "project": "System Threat Forecaster",
    "dataset": "Microsoft Malware Prediction",
    "total_samples": 100000,
    "training_samples": 80000,
    "validation_samples": 20000,
    "features": 75,
    "last_updated": "2025-12-18 HH:MM:SS"
  },
  "preprocessing": {
    "numeric_features": 47,
    "categorical_features": 28,
    "scaling": "StandardScaler",
    "encoding": "LabelEncoder",
    "imputation": {...}
  },
  "ml_models": {
    "<model_name>": {
      "training_accuracy": 0.XXXX,
      "validation_accuracy": 0.XXXX,
      "precision": 0.XXXX,
      "recall": 0.XXXX,
      "f1_score": 0.XXXX,
      "hyperparameters": {...},
      "trained_at": "YYYY-MM-DD HH:MM:SS"
    }
  },
  "dl_models": {
    "<model_name>": {
      "architecture": "ModelClass",
      "validation_accuracy": 0.XXXX,
      "best_val_loss": 0.XXXX,
      "total_parameters": NNNN,
      "trainable_parameters": NNNN,
      "hyperparameters": {...},
      "training": {...},
      "trained_at": "YYYY-MM-DD HH:MM:SS"
    }
  },
  "best_overall": {
    "model": "model_name",
    "accuracy": 0.XXXX,
    "type": "ML" or "DL"
  }
}
```

#### Usage:
```python
import json

# Load performance data
with open('results/model_performance.json', 'r') as f:
    perf_data = json.load(f)

# Get LightGBM hyperparameters
lgbm_params = perf_data['ml_models']['lightgbm']['hyperparameters']

# Get best model info
best_model = perf_data['best_overall']['model']
best_acc = perf_data['best_overall']['accuracy']

# Get all DL model accuracies
dl_accuracies = {
    name: model['validation_accuracy'] 
    for name, model in perf_data['dl_models'].items()
}
```

### model_comparison.csv
**Detailed training log** - Contains complete training history with timestamps, metrics, and hyperparameters for each training run.

#### Columns:
- `model_name`: Name of the model
- `timestamp`: Training timestamp
- `train_accuracy`: Training set accuracy
- `val_accuracy`: Validation set accuracy  
- `val_report`: Classification report (precision, recall, F1 per class)
- `hyperparams`: Complete hyperparameter configuration

### Confusion Matrix & Training History PNGs
Visual results for each model:
- `<model_name>_confusion_matrix.png`: Confusion matrix visualization
- `<model_name>_history.png`: Training/validation loss and accuracy curves (DL models only)

## Updating Performance Data

### ML Models
Performance data is automatically updated when running:
```bash
python system-threat-forecaster-ml.py
```

### DL Models  
Performance data is automatically updated when running:
```bash
python system-threat-forecaster-dl.py
```

## Integration Points

This data is used in:

1. **Web Application** (`next/app/models/page.tsx`)
   - Display model specifications
   - Show performance metrics
   - Compare model architectures

2. **Project Reports** (`report/`)
   - Include training results
   - Document hyperparameters
   - Show performance comparisons

3. **Presentations** (`beamer/`)
   - Display key metrics
   - Show best model results
   - Compare approaches

4. **Prediction API** (`predict.py`)
   - Reference model configurations
   - Validate input features
   - Log prediction metadata
