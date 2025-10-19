# Scripts/tuning.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

def load_data(features_path, target_path):
    """Features va target CSV dan yuklash"""
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    return X, y

def random_search_tuning(X_train, y_train):
    """Random Forest uchun RandomizedSearchCV tuning"""
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,  # 30 random parametrlar
        cv=5,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=2
    )
    random_search.fit(X_train, y_train)
    return random_search

def evaluate_model(model, X_test, y_test):
    """Modelni test to'plamida baholash"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Tuned model baholash natijalari:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")

def save_model(model, path):
    """Modelni saqlash"""
    joblib.dump(model, path)
    print(f"Model '{path}' ga saqlandi.")

if __name__ == "__main__":
    features_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\selected_features.csv"
    target_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\target_num_orders.csv"
    model_save_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Models\tuned_model.pkl"

    # Ma'lumotlarni yuklash
    X, y = load_data(features_path, target_path)

    # Train-test bo'lish
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomizedSearchCV bilan tuning qilish
    random_search = random_search_tuning(X_train, y_train)

    print(f"Eng yaxshi parametrlar:\n{random_search.best_params_}")

    # Testda baholash
    evaluate_model(random_search.best_estimator_, X_test, y_test)

    # Modelni saqlash
    save_model(random_search.best_estimator_, model_save_path)
