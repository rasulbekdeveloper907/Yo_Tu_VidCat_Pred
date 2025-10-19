from sklearn.model_selection import GridSearchCV
from models import get_models

def tune_model(X_train, y_train, model_name):
    """
    Berilgan model nomi uchun GridSearchCV yordamida eng yaxshi parametrlarni qidiradi.

    Args:
        X_train (pd.DataFrame): O'quv ma'lumotlari (features)
        y_train (pd.Series): O'quv ma'lumotlari (target)
        model_name (str): Tunelash uchun model nomi

    Returns:
        best_estimator_: eng yaxshi topilgan model obyekti
        best_params_: eng yaxshi parametrlar
        best_score_: eng yaxshi baho (R2 ko'rsatkich)
    """

    models = get_models()
    if model_name not in models:
        raise ValueError(f"Model {model_name} mavjud emas. Modellar ro'yxati: {list(models.keys())}")

    model = models[model_name]

    # Hyperparametrlar lug'ati (misol uchun)
    param_grids = {
        "Linear Regression": {},
        "Decision Tree": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "SVR": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    }

    param_grid = param_grids.get(model_name, {})
    if not param_grid:
        print(f"{model_name} modeli uchun tuning parametrlari aniqlanmagan. Asl model bilan ishlanmoqda.")
        model.fit(X_train, y_train)
        return model, None, None

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
