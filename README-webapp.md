# System Threat Forecaster - Web Application

Interactive web interface for malware detection using ML and Deep Learning models.

## Architecture

```
Training Phase (Python)
├── system-threat-forecaster-modular.py  → Train 7 ML models
├── system-threat-forecaster-dl.py       → Train 4 DL models
└── save_models.py                       → Package models for deployment

Inference Phase (Next.js + Python)
├── Next.js App (Frontend + API Routes)
├── predict.py (Python script)           → Load models & predict
└── saved_models/
    ├── ml_models.pkl                    → All ML models
    ├── dl_models.pth                    → All DL models
    └── preprocessors.pkl                → Scalers & encoders
```

## Setup Instructions

### 1. Train Models

First, train all models using the Python scripts:

```bash
# Train ML models (LightGBM, Random Forest, etc.)
python system-threat-forecaster-modular.py

# Train DL models (Attention Net, Residual Net, etc.)
python system-threat-forecaster-dl.py

# Package models for web app
python save_models.py
```

This will create a `saved_models/` directory with all trained models.

### 2. Install Dependencies

```bash
# Python dependencies (if not already installed)
pip install -r requirements.txt

# Next.js dependencies
cd next
npm install
```

### 3. Run Web App

```bash
cd next
npm run dev
```

Visit `http://localhost:3000`

## How It Works

### Training Phase

1. **ML Models** (scikit-learn):
   - LightGBM (best: 63% accuracy)
   - Random Forest
   - AdaBoost
   - Decision Tree
   - Naive Bayes
   - Logistic Regression
   - SGD Classifier
   - Saved as `.pkl` files using joblib

2. **DL Models** (PyTorch):
   - Attention Network (best: 61.73% accuracy)
   - Deep MLP
   - Residual Network
   - FT-Transformer
   - Saved as `.pth` files with torch.save()

3. **Preprocessors**:
   - StandardScaler for numerical features
   - LabelEncoders for categorical features
   - Saved as `.pkl` files

### Inference Phase

1. **User loads sample** → `/api/load-sample`
   - Reads from `train.csv`
   - Randomly selects one sample
   - Returns 76 features + actual label

2. **User requests predictions** → `/api/predict`
   - Next.js API route receives features
   - Spawns Python process with `predict.py`
   - Python script:
     - Loads saved models
     - Preprocesses features
     - Runs all 11 models
     - Returns predictions with confidence scores
   - API route returns JSON to frontend

3. **Frontend displays results**
   - Shows all 76 features in organized view
   - Displays predictions from all 11 models
   - Compares results with actual label
   - Highlights best model and correct predictions

## File Structure

```
qip-project-stf/
├── system-threat-forecaster-modular.py  # ML training script
├── system-threat-forecaster-dl.py       # DL training script
├── predict.py                           # Prediction script for web app
├── save_models.py                       # Model packaging script
├── saved_models/                        # Trained models (created by save_models.py)
│   ├── ml_models.pkl
│   ├── dl_models.pth
│   └── preprocessors.pkl
├── kaggle/
│   └── input/System-Threat-Forecaster/
│       ├── train.csv                    # 100K samples for training
│       └── test.csv                     # Test data
└── next/                                # Next.js web app
    ├── app/
    │   ├── page.tsx                     # Homepage
    │   ├── predict/page.tsx             # Prediction interface
    │   └── api/
    │       ├── load-sample/route.ts     # Load random sample
    │       └── predict/route.ts         # Get predictions
    ├── components/
    │   ├── navigation.tsx
    │   ├── feature-display.tsx
    │   └── model-comparison.tsx
    └── package.json
```

## API Endpoints

### GET `/api/load-sample`

Returns a random sample from the training dataset.

**Response:**
```json
{
  "features": {
    "AvSigVersion": 123456,
    "AppVersion": 789,
    ...
  },
  "target": 1,
  "sampleIndex": 42398
}
```

### POST `/api/predict`

Get predictions from all 11 models.

**Request:**
```json
{
  "features": {
    "AvSigVersion": 123456,
    "AppVersion": 789,
    ...
  }
}
```

**Response:**
```json
{
  "ml_models": {
    "lightgbm": {
      "prediction": 1,
      "confidence": 0.87
    },
    "random_forest": {
      "prediction": 1,
      "confidence": 0.82
    },
    ...
  },
  "dl_models": {
    "attention_net": {
      "prediction": 1,
      "confidence": 0.79
    },
    ...
  }
}
```

## Deployment

For production deployment at stf.milav.in:

1. **Build Next.js app:**
   ```bash
   cd next
   npm run build
   ```

2. **Start production server:**
   ```bash
   npm start
   ```

3. **Configure nginx** (reverse proxy):
   ```nginx
   server {
       listen 80;
       server_name stf.milav.in;
       
       location / {
           proxy_pass http://localhost:3000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
           proxy_set_header Host $host;
           proxy_cache_bypass $http_upgrade;
       }
   }
   ```

4. **Ensure Python dependencies** are available on server:
   ```bash
   pip install -r requirements.txt
   ```

5. **Copy saved models** to server in the project root directory

## Technologies

- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **UI Components**: shadcn/ui, Radix UI
- **Backend**: Next.js API Routes (Node.js)
- **ML/DL**: Python, scikit-learn, PyTorch
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib, torch.save()

## Performance

- **Best ML Model**: LightGBM - 63.0% accuracy
- **Best DL Model**: Attention Network - 61.73% accuracy
- **Inference Time**: ~100-200ms for all 11 models
- **Dataset**: 100,000 samples, 76 features

## Notes

- The web app uses the same preprocessing pipeline as training
- All models are loaded once when predict.py is called
- Predictions run on CPU (fast enough for web use)
- For GPU inference, ensure PyTorch CUDA/MPS support is configured
