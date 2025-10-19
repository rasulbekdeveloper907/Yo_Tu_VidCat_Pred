
import pandas as pd

class Cleaner:
    def __init__(self, df):
        self.df = df

    def tozala(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self

    def get_df(self):
        return self.df
    

    
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Encoder:
    def __init__(self, df):
        self.df = df
        self.encoder = LabelEncoder()

    def encodla(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                else:
                    self.df[col] = self.encoder.fit_transform(self.df[col])
        return self

    def get_df(self):
        return self.df 
    
    import pandas as pd
from sklearn.preprocessing import StandardScaler

class Scaler:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()

    def scaling_qil(self):
        numeric_cols = self.df.select_dtypes(include=['int64']).columns
        self.df[numeric_cols] = pd.DataFrame(
            self.scaler.fit_transform(self.df[numeric_cols]),
            columns=numeric_cols,
            index=self.df.index
        )
        return self

    def get_df(self):
        return self.df
