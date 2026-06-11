import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("🔄 Step 1: Loading Dataset...")
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        print(f"✅ Dataset loaded with {df.shape[0]} records.")
    except FileNotFoundError:
        print("❌ Error: 'Crop_recommendation.csv' not found. Please place it in this folder.")
        return

    print("\n🔄 Step 2: Splitting Features and Target Labels...")
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\n🔄 Step 3: Standardizing Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n🔄 Step 4: Training Random Forest Model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("\n🔄 Step 5: Evaluating Engine Accuracy...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Training Accuracy Score: {accuracy * 100:.2f}%")

    print("\n🔄 Step 6: Exporting Production Artifacts...")
    joblib.dump(model, "crop_recommendation_model.pkl")
    joblib.dump(scaler, "soil_features_scaler.pkl")
    print("💾 Saved 'crop_recommendation_model.pkl' and 'soil_features_scaler.pkl' successfully!")

if __name__ == "__main__":
    main()