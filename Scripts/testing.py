# Scripts/testing.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data(features_path, target_path):
    """Features va target CSV dan yuklash"""
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    return X, y

def evaluate_model(model_path, X_test, y_test):
    """Saqlangan modelni yuklash, testda baholash"""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model testi natijalari:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")

if __name__ == "__main__":
    features_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\selected_features.csv"
    target_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\target_num_orders.csv"
    model_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Models\best_model.pkl"  # o'zingiz saqlagan model yo'li
    
    X, y = load_data(features_path, target_path)
    
    # Agar test to'plam kerak bo'lsa, train_test_split qilish mumkin, yoki alohida test fayli bilan ishlash
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    evaluate_model(model_path, X_test, y_test)
