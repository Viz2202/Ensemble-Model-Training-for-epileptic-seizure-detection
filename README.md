# 🧠 Epileptic Seizure Detection using Machine Learning & Deep Learning

This project develops and compares multiple machine learning and deep learning models to classify epileptic seizures from raw EEG signals. After evaluating four standalone models, a **4-model majority-vote ensemble** is created and deployed with a **FastAPI** web application for real-time inference.

## 📌 Features

- Converts a 5-class EEG dataset into a **binary Seizure vs. Non-Seizure** problem  
- Handles **1:4 class imbalance** using class weights  
- Trains **four independent models** on raw EEG sequences  
- Builds a robust **majority-vote ensemble**  
- Deploys the final system with **FastAPI + HTML/JS frontend**  
- Provides real-time seizure prediction from raw EEG vector input (178 values)

## 📊 Model Performance (Binary Classification)

| Model        | Architecture               | Accuracy | F1-Score (Seizure) | Recall (Seizure) |
|--------------|-----------------------------|----------|--------------------|------------------|
| XGBoost      | Gradient Boosted Trees     | 97.61%   | 0.94               | 0.92             |
| SimpleRNN    | Deep Learning (RNN)        | 90.13%   | 0.71               | 0.60             |
| Stacked LSTM | Deep Learning (RNN)        | 96.78%   | 0.92               | 0.92             |
| CNN-LSTM     | Deep Learning (Hybrid)     | 98.57%   | 0.97               | 0.96             |
| **Ensemble** | 4-Model Majority Vote      | **98.83%** | **0.97**           | **0.96**         |

## 🚀 How to Run the Web Application

### 1. Prerequisites
- Python **3.8+**
- All trained model files in the `models/` directory

### 2. Installation
```
git clone YOUR_REPO_URL
cd YOUR_PROJECT_FOLDER
pip install -r requirements.txt
```

### 3. Start the FastAPI Server
```
python main.py
```

### 4. Use the Application
- Visit **http://127.0.0.1:8000**
- Paste comma‑separated EEG values (178 numbers)
- View model predictions + ensemble output

## 🧠 Dataset Overview

- Source: [Kaggle](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)  
- Original: 11,500 samples, 178 features, 5 classes  
- Converted to binary (Seizure vs. Non-Seizure)

## 🔧 Preprocessing

- StandardScaler
- Reshaping to `(N, 178, 1)` for DL models
- One-hot encoding
- Class imbalance handled with class weights

## 🏗 Model Architectures

### XGBoost
- Baseline model treating 178 features independently

### SimpleRNN
```
Input → SimpleRNN(64) → Dropout → Dense → Dense(sigmoid)
```

### Stacked LSTM
```
Input → LSTM(100, rs=True) → Dropout → LSTM(100) → Dropout → Dense → Dense(sigmoid)
```

### CNN‑LSTM (Best Individual Model)
```
Input → Conv1D → MaxPool → LSTM(100) → Dropout → Dense → Dense(sigmoid)
```

## 🤝 Ensemble Model

- Combines all 4 models via majority voting  
- Most robust & accurate predictor

