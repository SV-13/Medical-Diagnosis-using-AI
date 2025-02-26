import pandas as pd
import joblib
import os

# Define directories
dataset_dir = os.path.join(os.getcwd(), "datasets")
models_dir = os.path.join(os.getcwd(), "models")

# Dataset target column mappings
datasets = {
    "diabetes_data.csv": "Outcome",
    "heart_disease_data.csv": "target",
    "hypothyroid.csv": "binaryClass",
    "parkinson_data.csv": "status",
    "prepocessed_hypothyroid.csv": "binaryClass",
    "prepocessed_lungs_data.csv": "LUNG_CANCER",
    "survey lung cancer.csv": "LUNG_CANCER"
}

print("Starting testing script...\n")

# Loop through all datasets for testing
for dataset, target_col in datasets.items():
    file_path = os.path.join(dataset_dir, dataset)
    dataset_name = os.path.splitext(dataset)[0]  # Extract dataset name without .csv
    
    scaler_path = os.path.join(models_dir, f"{dataset_name}_scaler.pkl")
    feature_names_path = os.path.join(models_dir, f"{dataset_name}_feature_names.pkl")
    
    # Check if dataset exists
    if not os.path.exists(file_path):
        print(f"❌ Error: Dataset not found at {file_path}. Skipping...\n")
        continue
    
    # Check if feature names file exists
    if not os.path.exists(feature_names_path):
        print(f"❌ Error: Feature names file not found at {feature_names_path}. Skipping...\n")
        continue
    
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"✅ {dataset} loaded successfully! Shape: {df.shape}")
    
    # Load trained feature names
    trained_feature_names = joblib.load(feature_names_path)
    print(f"✅ Feature names loaded for {dataset}!")
    
    # Ensure target column exists
    if target_col not in df.columns:
        print(f"❌ Error: Target column '{target_col}' not found in {dataset}! Skipping...\n")
        continue
    
    # Drop target column
    X = df.drop(columns=[target_col])
    print(f"✅ Features extracted! Shape: {X.shape}")
    
    # Convert categorical values to numerical (if any)
    X = pd.get_dummies(X)
    
    # Add missing columns (if any)
    for col in trained_feature_names:
        if col not in X.columns:
            X[col] = 0  # Fill missing columns
    
    # Ensure correct column order
    X = X[trained_feature_names]
    
    # Check if scaler exists
    if not os.path.exists(scaler_path):
        print(f"❌ Error: Scaler file not found at {scaler_path}. Skipping...\n")
        continue
    
    # Load and apply scaler
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)
    print(f"✅ {dataset} successfully scaled!\n")

print("🎉 All datasets tested successfully!")
