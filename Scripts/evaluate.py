# Scripts/evaluate.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(features_path, target_path):
    """Features va target CSV dan yuklash"""
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Ma'lumotlarni train-test ga bo'lish"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_models(X_train, X_test, y_train, y_test):
    """Turli regressiya modellari uchun trening, baholash va natijalarni DataFrame ga yig'ish"""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })
        
    results_df = pd.DataFrame(results)
    return results_df

def plot_results(results_df):
    """Baholash natijalarini grafikda chizish"""
    results_long = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_long, x="Model", y="Score", hue="Metric")
    plt.title("Regressiya modellari metrikalari solishtirish")
    plt.xticks(rotation=45)
    plt.ylabel("Qiymat")
    plt.xlabel("Model nomi")
    plt.legend(title="Metriа")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    features_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\selected_features.csv"
    target_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\target_num_orders.csv"
    
    X, y = load_data(features_path, target_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    results_df = evaluate_models(X_train, X_test, y_train, y_test)
    
    print(results_df)
    plot_results(results_df)
