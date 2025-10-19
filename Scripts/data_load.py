# Scripts/data_load.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt

def load_data(features_path, target_path):
    """Features va target CSV dan yuklash"""
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()  # Series sifatida olish
    return X, y

def train_model(X, y):
    """Ma'lumotlarni train-test ga bo'lish va modelni o'qitish"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse:.6f}")
    
    return model, X_train, X_test, y_train, y_test

def shap_analysis(model, X_train, X_test):
    """SHAP qiymatlarni hisoblash va vizualizatsiya qilish"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Summary plot (beeswarm)
    shap.summary_plot(shap_values, X_test, plot_type="dot")
    
    # Summary plot (bar)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    
    # Dependence plot (agar 'checkout_price' mavjud bo'lsa)
    if 'checkout_price' in X_test.columns:
        shap.dependence_plot("checkout_price", shap_values, X_test)
    
    # Force plot (birinchi test namunasi uchun)
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
    shap.save_html("force_plot.html", force_plot)
    
    # Waterfall plot (birinchi test namunasi uchun)
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value, 
                                         data=X_test.iloc[0]))
    
    # Decision plot (birinchi 100 namunalar uchun)
    shap.decision_plot(explainer.expected_value, shap_values[:100], X_test.iloc[:100])
    
    plt.show()

if __name__ == "__main__":
    features_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\selected_features.csv"
    target_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\target_num_orders.csv"
    
    X, y = load_data(features_path, target_path)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    shap_analysis(model, X_train, X_test)
