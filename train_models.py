import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Define dataset directory
dataset_dir = os.path.join(os.getcwd(), "datasets")

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

# Create 'models' directory if not exists
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

# Loop through datasets and train models
for dataset, target_col in datasets.items():
    dataset_path = os.path.join(dataset_dir, dataset)

    if not os.path.exists(dataset_path):
        print(f"❌ {dataset} not found! Skipping...")
        continue

    try:
        # Load dataset
        df = pd.read_csv(dataset_path)

        # Ensure target column exists
        if target_col not in df.columns:
            print(f"❌ Error: Target column '{target_col}' not found in {dataset}. Skipping...")
            continue

        print(f"\n🚀 Training model for {dataset}...")

        # Data Preprocessing
        X = df.drop(columns=[target_col])  # Features (exclude target)
        y = df[target_col]  # Target variable

        # Convert categorical values to numerical (if any)
        X = pd.get_dummies(X)

        # Split dataset (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features (important for SVM & Logistic Regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 🔹 Remove '.csv' from dataset name before saving feature names
        dataset_name = os.path.splitext(dataset)[0]  # Removes the ".csv" extension
        
        # 🔹 Save feature names used during training
        feature_names_path = os.path.join(models_dir, f"{dataset_name}_feature_names.pkl")
        joblib.dump(X_train.columns.tolist(), feature_names_path)
        print(f"✅ Feature names saved at: {feature_names_path}")

        # 🔹 Save the scaler for consistent feature scaling
        scaler_path = os.path.join(models_dir, f"{dataset_name}_scaler.pkl")  # FIXED Filename Issue
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved at: {scaler_path}")

        print(f"Training {dataset} with {X.shape[1]} features...")  

        # Train SVM Model
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)

        # Train Logistic Regression Model
        logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        logistic_model.fit(X_train_scaled, y_train)

        # Train Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)  # No need to scale for RandomForest

        # Evaluate Models
        svm_acc = accuracy_score(y_test, svm_model.predict(X_test_scaled))
        logistic_acc = accuracy_score(y_test, logistic_model.predict(X_test_scaled))
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

        print(f"✅ {dataset} - SVM Accuracy: {svm_acc:.4f}")
        print(f"✅ {dataset} - Logistic Regression Accuracy: {logistic_acc:.4f}")
        print(f"✅ {dataset} - Random Forest Accuracy: {rf_acc:.4f}")

    except Exception as e:
        print(f"❌ Error processing {dataset}: {e}")

print("\n🎉 All models trained and saved successfully!")
