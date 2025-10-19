from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

def get_models():
    """
    Turli regressiya modellarini lug'at ko'rinishida qaytaradi.
    Keyinchalik bu modellar tuning va treningda ishlatiladi.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }
    return models
