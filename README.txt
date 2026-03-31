# LSTM-Based Channel Correlation Coefficient (ρ) Prediction

This project implements a deep learning solution to predict the correlation coefficient (ρ) in a Gauss-Markov AR(1) channel model using LSTM networks. The model learns to estimate ρ from sequences of complex channel states without having ρ explicitly in the input features.

## 📋 Project Overview

The Gauss-Markov AR(1) channel model is defined as:

h(t) = ρ * h(t-1) + √(σ²(1-ρ²)) * n(t)

where:
- `h(t)` is the complex channel state at time t
- `ρ` is the correlation coefficient (target variable)
- `σ²` is the noise variance
- `n(t)` is complex Gaussian noise

The project consists of two main components:
1. **Dataset Generation**: Creates synthetic channel sequences with known ρ values
2. **LSTM Training**: Trains a neural network to predict ρ from channel sequences

## 📁 Project Structure


project/
├── part1_generate_dataset.py # Dataset generation script
├── part2_train_model.py # LSTM training and evaluation script
├── README.md # This file
├── requirements.txt # Python dependencies
├── dataset/ # Generated dataset (created after running part1)
│ ├── rho_data_X.csv
│ ├── rho_data_y.csv
│ ├── rho_data_metadata.csv
│ └── rho_data_shape_info.csv
└── results/ # Training results (created after running part2)
├── training_history.png
├── predictions_scatter.png
├── error_histogram.png
├── residual_plot.png
├── best_model.keras
└── final_model.keras


## 🔧 Requirements

- Python 3.10 (TensorFlow doesn't support Python 3.11+)
- Required packages:

numpy==1.24.3
tensorflow==2.10.0
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.7.1


## 🚀 Installation and Setup

### 1. Clone/Download the Project
```bash
git clone <your-repository-url>
cd <project-folder>

Create a Virtual Environment (Recommended)
Windows (PowerShell):

powershell
# Create virtual environment with Python 3.10
py -3.10 -m venv tf_env

Install Dependencies:

# Upgrade pip
python -m pip install --upgrade pip

# Install packages
pip install numpy==1.24.3 tensorflow==2.10.0 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1

Dataset Generation
Generate the synthetic dataset:

python part1_generate_dataset.py

Configuration parameters (can be modified in the script):

N_SAMPLES: Number of sequences (default: 10000)

N_ANTENNAS: Number of antennas (default: 4)

SEQ_LENGTH: Sequence length (default: 10)

RHO_RANGE: Range of ρ values (default: 0.6 to 0.95)

NOISE_VAR: Noise variance (default: 0.05)

Output:

5000 samples × 10 timesteps × 8 features (4 real + 4 imaginary parts)

Target: ρ values between 0.6 and 0.95

 Model Training
Train the LSTM model:

python part2_train_model.py

Model Architecture:

LSTM layer with 64 units

Dense layer with 32 units (ReLU activation)

Dropout layer (0.2)

Output layer (linear activation)

Training Configuration:

Epochs: 100 (exactly)

Batch size: 64

Optimizer: Adam (learning rate: 0.001)

Loss function: MSE

Train/Val/Test split: 70%/15%/15%

📈 Results and Visualization
After training, the following files are generated in the results/ folder:

1. Training History (training_history.png)
Loss (MSE) curves for training and validation

MAE curves for training and validation

2. Prediction Scatter Plot (predictions_scatter.png)
True ρ vs Predicted ρ

Perfect prediction line (red dashed)

R² score displayed

3. Error Histogram (error_histogram.png)
Distribution of prediction errors

Mean and standard deviation of errors

4. Residual Plot (residual_plot.png)
Residuals vs Predicted values

Helps identify patterns in errors

5. Model Files
best_model.keras: Best model based on validation loss

final_model.keras: Model after 100 epochs

📊 Performance Metrics
The evaluation provides:

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² (R-squared score)

🔬 Technical Details
Data Generation Process
Random ρ sampled uniformly from specified range

Initial complex channel vector (unit variance)

AR(1) process generates sequence of channel states
Each state converted to real features (concatenated real and imaginary parts)

ρ stored as target variable

Preprocessing
Features: StandardScaler (zero mean, unit variance)

Target: StandardScaler (zero mean, unit variance)

Thank you :)
