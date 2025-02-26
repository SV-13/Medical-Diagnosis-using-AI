import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

# Define file paths
print("Starting script...")

file_path = "C:/Users/sujal/OneDrive/Documents/Projects/Medical diagnosis using AI/data/datasets/prepocessed_hypothyroid.csv"
scaler_path = "C:/Users/sujal/OneDrive/Documents/Projects/Medical diagnosis using AI/data/models/prepocessed_hypothyroid_scaler.pkl"

# Check if the dataset exists
if not os.path.exists(file_path):
    print(f"❌ Error: Dataset not found at {file_path}")
    exit()

# Load preprocessed dataset (correct feature set)
df = pd.read_csv(file_path)

print(f"✅ Dataset loaded successfully! Shape: {df.shape}")

# Drop target column if needed (ensure only input features are scaled)
if "binaryClass" in df.columns:
    X = df.drop(columns=["binaryClass"])
else:
    print("❌ Error: 'binaryClass' column not found in dataset!")
    print("Columns found:", df.columns.tolist())
    exit()

print(f"✅ Features extracted successfully! Shape: {X.shape}")

# Fit new scaler
scaler = StandardScaler()
scaler.fit(X)

print("✅ Scaler trained successfully!")

# Save the correctly trained scaler
joblib.dump(scaler, scaler_path)

print(f"✅ Scaler saved successfully at: {scaler_path}")
print("Scaler expects features:", scaler.n_features_in_)
