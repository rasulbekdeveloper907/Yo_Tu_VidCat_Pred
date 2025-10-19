import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def load_data(features_path, target_path):
    """Features va target CSV dan yuklash"""
    X = pd.read_csv(features_path)
    y = pd.read_csv(target_path).squeeze()
    return X, y

def load_model(model_path):
    """Saqlangan modelni yuklash"""
    model = joblib.load(model_path)
    print(f"Model '{model_path}' yuklandi.")
    return model

def evaluate_model(model, X_test, y_test):
    """Modelni test to'plamida baholash va metrikalarni chiqarish"""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model baholash natijalari:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R2: {r2:.6f}")

    return y_pred

def shap_analysis(model, X_train, X_test):
    """SHAP yordamida feature importance va boshqa grafiklar"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot (bar va beeswarm)
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.title("SHAP Summary Plot (Beeswarm)")
    plt.tight_layout()
    plt.show()

    # Misol uchun bir feature bo'yicha dependence plot
    if "checkout_price" in X_test.columns:
        shap.dependence_plot("checkout_price", shap_values, X_test)
    else:
        print("Eslatma: 'checkout_price' feature topilmadi, dependence plot chizilmadi.")

if __name__ == "__main__":
    features_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\selected_features.csv"
    target_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Data\feature_selection\target_num_orders.csv"
    model_path = r"C:\Users\Rasulbek907\Desktop\SML_2_Pr\Models\tuned_model.pkl"

    # Ma'lumotlarni yuklash
    X, y = load_data(features_path, target_path)

    # Train-test bo'lish (SHAP uchun train kerak)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelni yuklash
    model = load_model(model_path)

    # Modelni baholash
    y_pred = evaluate_model(model, X_test, y_test)

    # SHAP tahlili
    shap_analysis(model, X_train, X_test)
