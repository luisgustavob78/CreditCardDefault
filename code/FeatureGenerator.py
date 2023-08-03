import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

class DropCol(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data = data.drop(columns=self.features, axis=1)
        return data

class FeatureByFeature(BaseEstimator, TransformerMixin):

    def __init__(self, features_num, features_denom):
        self.features_num = features_num
        self.features_denom = features_denom

    def feature_calculator(self, data, f1, f2):
        for i in range(0, len(f1)):
            v1 = f1[i]
            v2 = f2[i]

            data[f"{v1}/{v2}"] = data[v1]/data[v2]

        return data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_final = X.copy()
        df_final = self.feature_calculator(df_final, self.features_num, self.features_denom)
        return df_final

class DiffFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        for i in range(1, len(self.features)):
            a = self.features[i-1]
            b = self.features[i]
            data[f"{b}_minus_{a}"] = b - a 

        return data

class Numeric2Buckets(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def binning(self, data, features):
        for f in features:

class FinalFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data = data[self.features]

        return data