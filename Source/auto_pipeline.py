import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class PlayerPipeline(BaseEstimator):   
    def __init__(self, df: pd.DataFrame, target: str, model=None, test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target = target
        self.model_algorithm = model
        self.model = None
        self.preprocessor = None
        self.test_size = test_size
        self.random_state = random_state

    def _prepare_features(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        num_col = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_col = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, num_col),
            ('cat', categorical_pipeline, cat_col)
        ])

        # 🔹 Train/Test ajratamiz
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y if len(np.unique(y)) > 1 else None
        )

        return self.X_train, self.y_train

    def fit(self):
        X_train, y_train = self._prepare_features()

        if self.model_algorithm is None:
            raise ValueError("No model specified. Pass a scikit-learn estimator.")

        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('estimator', self.model_algorithm)
        ])

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X=None):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if X is None:
            X = self.X_test  # default holda testdan bashorat oladi
        return self.model.predict(X)

    def score(self, X=None, y=None):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if X is None and y is None:
            X, y = self.X_test, self.y_test
        return self.model.score(X, y)
