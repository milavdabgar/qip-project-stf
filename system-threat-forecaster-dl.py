# %% [markdown]
"""
# System Threat Forecaster using Deep Learning Approach

This project implements deep learning models using PyTorch to predict malware infections.
The goal is to improve upon the 63% accuracy achieved by traditional ML methods (LightGBM).

Key Features:
1. Multiple Neural Network Architectures: MLP, Deep Networks, Residual Networks, Attention-based
2. Advanced Training Techniques: Learning rate scheduling, early stopping, dropout, batch normalization
3. PyTorch Implementation: GPU acceleration, efficient data loading, mixed precision training
4. Ensemble Methods: Model averaging, stacking for better predictions

Why Deep Learning:
- Can capture complex non-linear patterns
- Automatic feature learning from raw data
- Potential for better performance with proper architecture
- Transfer learning capabilities
"""

# %% [markdown]
"""
## 0. Setup and Installation

This section imports all necessary libraries for deep learning.
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Check for GPU availability - Support CUDA (NVIDIA) and MPS (Apple Silicon)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: CUDA")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: MPS (Apple Silicon GPU)")
    print("🚀 Apple Silicon GPU acceleration enabled! Training will be much faster.")
else:
    device = torch.device('cpu')
    print("Using device: CPU")
    print("Running on CPU - training will be slower but still effective")
    print("Tip: Consider using smaller batch sizes or fewer epochs if training is too slow")

# %% [markdown]
"""
## 1. Configuration

Deep learning specific configuration including:
- Network architectures
- Training hyperparameters
- Optimization settings
- Data augmentation options
"""

# %%
CONFIG = {
    # Data paths
    'data_path': {
        'train': './kaggle/input/System-Threat-Forecaster/train.csv',
        'test': './kaggle/input/System-Threat-Forecaster/test.csv',
        'submission': 'submission_dl.csv'
    },
    
    # Training parameters (auto-adjusted for CPU vs GPU)
    'batch_size': 512 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 256,
    'epochs': 100 if (torch.cuda.is_available() or torch.backends.mps.is_available()) else 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 15,
    
    # Model architecture
    'hidden_dims': [256, 128, 64, 32],  # Hidden layer dimensions
    'dropout_rate': 0.3,
    'use_batch_norm': True,
    'activation': 'relu',  # relu, gelu, elu
    
    # Models to train
    'models_to_train': {
        'simple_mlp': True,
        'deep_mlp': True,
        'residual_net': True,
        'attention_net': True,
        'wide_deep': True,
        'ft_transformer': True,  # State-of-the-art for tabular data
    },
    
    # Optimization
    'optimizer': 'adamw',  # adam, adamw, sgd
    'scheduler': 'reduce_on_plateau',  # reduce_on_plateau, cosine, onecycle
    'use_mixed_precision': torch.cuda.is_available(),  # Only on NVIDIA GPUs
    
    # Data settings
    'test_size': 0.2,
    'random_state': SEED,
    'handle_class_imbalance': True,
    
    # Other settings
    'save_best_model': True,
    'verbose': True,
}

# %% [markdown]
"""
## 2. Data Loading and Preprocessing

Similar to the ML version but optimized for PyTorch deep learning.
"""

# %%
def load_data(path=None):
    """Load dataset from CSV file."""
    if path is None:
        path = CONFIG['data_path']['train']
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

def preprocess_data(data, is_training=True, target_col='target', preprocessors=None):
    """
    Preprocess data for deep learning.
    
    Args:
        data: Input DataFrame
        is_training: Whether this is training data
        target_col: Name of target column
        preprocessors: Dictionary of fitted preprocessors
        
    Returns:
        X, y (if training), preprocessors
    """
    df = data.copy()
    
    # Separate features and target
    if is_training and target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Found {len(numeric_cols)} numerical and {len(categorical_cols)} categorical columns")
    
    # Initialize preprocessors
    if preprocessors is None:
        preprocessors = {
            'numeric_imputer': SimpleImputer(strategy='mean'),
            'categorical_imputer': SimpleImputer(strategy='most_frequent'),
            'scaler': StandardScaler(),
            'label_encoders': {}
        }
        
        # Create label encoders for categorical columns
        for col in categorical_cols:
            preprocessors['label_encoders'][col] = LabelEncoder()
    
    # Handle missing values and encode categorical features
    if len(numeric_cols) > 0:
        if is_training:
            X[numeric_cols] = preprocessors['numeric_imputer'].fit_transform(X[numeric_cols])
        else:
            X[numeric_cols] = preprocessors['numeric_imputer'].transform(X[numeric_cols])
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            # Impute missing values
            if is_training:
                X[col] = preprocessors['categorical_imputer'].fit_transform(
                    X[col].values.reshape(-1, 1)
                ).flatten()
            else:
                X[col] = preprocessors['categorical_imputer'].transform(
                    X[col].values.reshape(-1, 1)
                ).flatten()
            
            # Label encode
            if is_training:
                X[col] = preprocessors['label_encoders'][col].fit_transform(X[col].astype(str))
            else:
                # Handle unknown categories
                known_categories = set(preprocessors['label_encoders'][col].classes_)
                unknown_mask = ~X[col].astype(str).isin(known_categories)
                if unknown_mask.any():
                    X.loc[unknown_mask, col] = preprocessors['label_encoders'][col].classes_[0]
                X[col] = preprocessors['label_encoders'][col].transform(X[col].astype(str))
    
    # Scale all features
    if is_training:
        X_scaled = preprocessors['scaler'].fit_transform(X)
    else:
        X_scaled = preprocessors['scaler'].transform(X)
    
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    if is_training:
        return X.values, y, preprocessors
    else:
        return X.values, preprocessors

# %% [markdown]
"""
## 3. PyTorch Dataset and DataLoader

Custom dataset class for efficient data loading.
"""

# %%
class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data."""
    
    def __init__(self, X, y=None):
        """
        Args:
            X: Feature matrix (numpy array)
            y: Target vector (numpy array or None)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=None):
    """Create PyTorch DataLoaders for training and validation."""
    if batch_size is None:
        batch_size = CONFIG['batch_size']
    
    # Calculate class weights for imbalanced data
    if CONFIG['handle_class_imbalance']:
        unique, counts = np.unique(y_train, return_counts=True)
        class_weights = torch.FloatTensor(len(y_train) / (len(unique) * counts))
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    
    # Create datasets
    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    
    # Create dataloaders
    # num_workers=0 for MPS to avoid multiprocessing issues on macOS
    num_workers = 0  # Multiprocessing can cause issues with MPS on macOS
    use_pin_memory = torch.cuda.is_available()  # pin_memory only for CUDA, not MPS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, class_weights

# %% [markdown]
"""
## 4. Neural Network Architectures

Multiple architectures designed for tabular data:
1. Simple MLP: Basic feedforward network
2. Deep MLP: Deeper network with batch norm and dropout
3. Residual Network: Skip connections for better gradient flow
4. Attention Network: Self-attention for feature interactions
5. Wide & Deep: Combined memorization and generalization
"""

# %%
class SimpleMLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

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
        
        out += residual  # Skip connection
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
        # Add dimension for attention (batch, seq_len, features)
        x = x.unsqueeze(1)
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Remove added dimension
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

class WideAndDeep(nn.Module):
    """Wide & Deep Network combining linear and deep components."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(WideAndDeep, self).__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, 2)
        
        # Deep component
        deep_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        deep_layers.append(nn.Linear(prev_dim, 2))
        
        self.deep = nn.Sequential(*deep_layers)
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        return wide_out + deep_out  # Combine outputs

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data.
    State-of-the-art architecture from "Revisiting Deep Learning Models for Tabular Data" (2021).
    
    Key innovations:
    - Feature tokenization: each feature becomes a learnable token
    - Transformer encoder for feature interactions
    - CLS token for final prediction
    """
    
    def __init__(self, input_dim, d_token=192, n_blocks=3, attention_heads=8, 
                 ffn_factor=4, dropout_rate=0.1):
        super(FTTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_token = d_token
        
        # Feature tokenization: project each feature to d_token dimension
        self.feature_tokenizer = nn.Linear(1, d_token)
        
        # CLS token (for classification)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Positional embeddings for each feature
        self.feature_embeddings = nn.Parameter(torch.randn(input_dim, d_token))
        
        # Transformer encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=attention_heads,
            dim_feedforward=d_token * ffn_factor,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        
        # Output layer
        self.layer_norm = nn.LayerNorm(d_token)
        self.output = nn.Linear(d_token, 2)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Feature tokenization: (batch, features) -> (batch, features, d_token)
        x = x.unsqueeze(-1)  # (batch, features, 1)
        feature_tokens = self.feature_tokenizer(x)  # (batch, features, d_token)
        
        # Add feature embeddings
        feature_tokens = feature_tokens + self.feature_embeddings.unsqueeze(0)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, feature_tokens], dim=1)  # (batch, 1+features, d_token)
        
        # Transformer encoding
        tokens = self.transformer(tokens)
        
        # Use CLS token for classification
        cls_output = tokens[:, 0]  # (batch, d_token)
        cls_output = self.layer_norm(cls_output)
        
        # Final prediction
        output = self.output(cls_output)
        return output

# %% [markdown]
"""
## 5. Training Functions

Comprehensive training pipeline with:
- Early stopping
- Learning rate scheduling
- Mixed precision training
- Model checkpointing
"""

# %%
class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.compare = lambda a, b: a < b - min_delta
        else:
            self.compare = lambda a, b: a > b + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

def get_model(model_name, input_dim):
    """Get model instance by name."""
    
    if model_name == 'simple_mlp':
        return SimpleMLP(input_dim, hidden_dims=[256, 128, 64], dropout_rate=CONFIG['dropout_rate'])
    
    elif model_name == 'deep_mlp':
        return DeepMLP(
            input_dim,
            hidden_dims=CONFIG['hidden_dims'],
            dropout_rate=CONFIG['dropout_rate'],
            use_batch_norm=CONFIG['use_batch_norm']
        )
    
    elif model_name == 'residual_net':
        return ResidualNet(
            input_dim,
            hidden_dim=256,
            num_blocks=3,
            dropout_rate=CONFIG['dropout_rate'],
            use_batch_norm=CONFIG['use_batch_norm']
        )
    
    elif model_name == 'attention_net':
        return AttentionNet(
            input_dim,
            hidden_dim=256,
            num_blocks=2,
            num_heads=4,
            dropout_rate=CONFIG['dropout_rate']
        )
    
    elif model_name == 'wide_deep':
        return WideAndDeep(
            input_dim,
            hidden_dims=[256, 128, 64],
            dropout_rate=CONFIG['dropout_rate']
        )
    
    elif model_name == 'ft_transformer':
        return FTTransformer(
            input_dim,
            d_token=64,          # Further reduced for speed
            n_blocks=1,          # Minimal transformer (single block)
            attention_heads=2,   # Minimal attention heads
            ffn_factor=2,        # Keep feedforward efficient
            dropout_rate=0.1
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (supports CUDA and MPS)
        if scaler is not None and CONFIG['use_mixed_precision']:
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
            elif device.type == 'mps':
                with torch.amp.autocast('mps'):
                    output = model(data)
                    loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Calculate F1 score
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_targets

def train_model(model_name, X_train, y_train, X_val, y_val):
    """
    Train a deep learning model.
    
    Args:
        model_name: Name of the model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained model and metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_loader, val_loader, class_weights = create_dataloaders(
        X_train, y_train, X_val, y_val
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = get_model(model_name, input_dim).to(device)
    
    # Use fewer epochs for transformer to speed up training
    max_epochs = 50 if model_name == 'ft_transformer' else CONFIG['epochs']
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if CONFIG['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    elif CONFIG['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9, weight_decay=CONFIG['weight_decay'])
    
    # Learning rate scheduler
    if CONFIG['scheduler'] == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif CONFIG['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    else:
        scheduler = None
    
    # Mixed precision scaler (supports CUDA and MPS)
    if CONFIG['use_mixed_precision']:
        if device.type == 'cuda':
            scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (faster on NVIDIA GPUs)")
        elif device.type == 'mps':
            scaler = torch.amp.GradScaler('mps')
            print("Using mixed precision training (faster on Apple Silicon)")
        else:
            scaler = None
            print("Using standard precision training (CPU)")
    else:
        scaler = None
        print("Mixed precision disabled")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'], mode='max')
    
    # Training loop
    best_val_acc = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(max_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Update scheduler
        if scheduler is not None:
            if CONFIG['scheduler'] == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Print progress
        if CONFIG['verbose'] and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
            if optimizer.param_groups[0]['lr'] != CONFIG['learning_rate']:
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if CONFIG['save_best_model']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"models/{model_name}_best_{timestamp}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                }, model_path)
        
        # Early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Final evaluation
    _, final_acc, final_f1, y_pred, y_true = validate(model, val_loader, criterion, device)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Malware', 'Malware']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Malware', 'Malware'],
                yticklabels=['No Malware', 'Malware'])
    plt.title(f'Confusion Matrix - {model_name.replace("_", " ").title()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot training history
    plot_training_history(history, model_name)
    
    metrics = {
        'best_val_acc': best_val_acc,
        'final_val_acc': final_acc,
        'final_val_f1': final_f1,
        'history': history
    }
    
    return model, metrics

def plot_training_history(history, model_name):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1 Score
    axes[2].plot(history['val_f1'], label='Val F1', color='green')
    axes[2].set_title('F1 Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.suptitle(f'Training History - {model_name.replace("_", " ").title()}')
    plt.tight_layout()
    plt.savefig(f'results/{model_name}_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
"""
## 6. Prediction and Submission

Generate predictions on test data.
"""

# %%
def predict(model, X_test):
    """Generate predictions on test data."""
    model.eval()
    
    # Create test dataset
    test_dataset = TabularDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'] * 2, shuffle=False)
    
    predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            if isinstance(data, list):
                data = data[0]
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    return np.array(predictions)

def generate_submission(model, X_test, output_path=None):
    """Generate submission file."""
    if output_path is None:
        output_path = CONFIG['data_path']['submission']
    
    print(f"\n{'='*60}")
    print("Generating Submission")
    print(f"{'='*60}")
    
    predictions = predict(model, X_test)
    
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'target': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Prediction distribution:")
    print(submission['target'].value_counts(normalize=True))
    
    return submission

# %% [markdown]
"""
## 7. Main Pipeline

Complete pipeline for deep learning approach.
"""

# %%
def run_dl_pipeline():
    """Run the complete deep learning pipeline."""
    print("="*80)
    print("SYSTEM THREAT FORECASTER - DEEP LEARNING APPROACH")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    train_data = load_data(CONFIG['data_path']['train'])
    test_data = load_data(CONFIG['data_path']['test'])
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X, y, preprocessors = preprocess_data(train_data)
    X_test, _ = preprocess_data(test_data, is_training=False, preprocessors=preprocessors)
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train models
    print("\n4. Training models...")
    trained_models = {}
    best_model = None
    best_acc = 0
    best_model_name = None
    
    for model_name, should_train in CONFIG['models_to_train'].items():
        if should_train:
            model, metrics = train_model(model_name, X_train, y_train, X_val, y_val)
            trained_models[model_name] = (model, metrics)
            
            if metrics['best_val_acc'] > best_acc:
                best_acc = metrics['best_val_acc']
                best_model = model
                best_model_name = model_name
    
    # Compare models
    if len(trained_models) > 1:
        print("\n5. Model Comparison:")
        comparison_data = []
        for model_name, (model, metrics) in trained_models.items():
            comparison_data.append({
                'Model': model_name,
                'Best Val Acc': f"{metrics['best_val_acc']:.2f}%",
                'Final Val Acc': f"{metrics['final_val_acc']:.2f}%",
                'Final Val F1': f"{metrics['final_val_f1']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    # Generate submission
    if best_model is not None:
        print(f"\n6. Generating submission with best model: {best_model_name}")
        submission = generate_submission(best_model, X_test)
    
    # Step 7: Save models for web app deployment
    print("\n7. Saving Models for Web App")
    print("="*80)
    
    os.makedirs('saved_models', exist_ok=True)
    
    # Save all DL models in PyTorch standard format (.pth)
    if trained_models:
        # Create a checkpoint with all models and metadata
        dl_checkpoint = {
            'models': {},
            'metadata': {
                'best_model': best_model_name,
                'best_accuracy': best_acc,
                'input_dim': X_train.shape[1],
                'num_classes': 2,
                'device': str(device)
            }
        }
        
        for model_name, (model, metrics) in trained_models.items():
            dl_checkpoint['models'][model_name] = {
                'model': model,
                'state_dict': model.state_dict(),
                'best_val_acc': metrics['best_val_acc'],
                'architecture': model.__class__.__name__
            }
        
        # Save with torch.save() - industry standard for PyTorch
        torch.save(dl_checkpoint, 'saved_models/dl_models.pth')
        print(f"✓ Saved {len(trained_models)} DL models to saved_models/dl_models.pth")
        
        # Calculate size
        size_mb = os.path.getsize('saved_models/dl_models.pth') / (1024 * 1024)
        print(f"  Model file size: {size_mb:.2f} MB")
        print(f"  Best model: {best_model_name} ({best_acc:.2f}% accuracy)")
    
    # Save preprocessors if not already saved by ML script
    preprocessors_path = 'saved_models/preprocessors.pkl'
    if not os.path.exists(preprocessors_path):
        joblib.dump(preprocessors, preprocessors_path)
        print(f"✓ Saved preprocessors to {preprocessors_path}")
    else:
        print(f"ℹ Preprocessors already exist at {preprocessors_path}")
    
    print("\nAll models saved successfully!")
    print("\nDeployment files ready:")
    for file in os.listdir('saved_models'):
        size = os.path.getsize(f'saved_models/{file}') / (1024 * 1024)
        print(f"  ✓ {file} ({size:.2f} MB)")
    
    print("\nNext steps:")
    print("  1. Test predictions: python predict.py")
    print("  2. Run web app: cd next && npm run dev")
    print("  3. Visit: http://localhost:3000/predict")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print(f"Best Model: {best_model_name} with {best_acc:.2f}% validation accuracy")
    print("="*80)
    
    return trained_models, best_model, best_model_name, preprocessors

# %% [markdown]
"""
## 8. Execution

Run the pipeline when script is executed.
"""

# %%
if __name__ == "__main__":
    trained_models, best_model, best_model_name, preprocessors = run_dl_pipeline()
