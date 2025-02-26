import pandas as pd
import os
import joblib

# Define dataset directory
dataset_dir = os.path.join(os.getcwd(), "datasets")

# List of datasets
datasets = [
    "diabetes_data.csv",
    "heart_disease_data.csv",
    "hypothyroid.csv",
    "parkinson_data.csv",
    "prepocessed_hypothyroid.csv",
    "prepocessed_lungs_data.csv",
    "survey lung cancer.csv"
]

for dataset in datasets:
    dataset_path = os.path.join(dataset_dir, dataset)
    
    # Check if the file exists
    if not os.path.exists(dataset_path):
        print(f"❌ Error: {dataset} not found!")
        continue

    try:
        # Load dataset
        df = pd.read_csv(dataset_path)

        # Print column names
        print(f"\n📂 {dataset} Columns:")
        print(df.columns.tolist())
    
    except Exception as e:
        print(f"❌ Error loading {dataset}: {e}")

# ====== ADD THIS PART TO CHECK SCALER ======

scaler_path = "C:/Users/sujal/OneDrive/Documents/Projects/Medical diagnosis using AI/data/models/hypothyroid.csv_scaler.pkl"

if os.path.exists(scaler_path):
    try:
        # Load the scaler
        scaler = joblib.load(scaler_path)

        # Check the number of expected features
        print("\n✅ Scaler loaded successfully!")
        print("Scaler expects features:", scaler.n_features_in_)  # Debugging step
    
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")

else:
    print(f"❌ Error: Scaler file not found at {scaler_path}")
