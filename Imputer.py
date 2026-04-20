"""
This class provides methods for handling missing values in a DataFrame.
It provides methods for:
-mean
-median
-mode
-custom imputation option

Structure:
-Caluculate missingness per column and print to user (count all nans and or nulls)
-Allow user to choose missingness threshold for dropping columns
-Allow user to choose imputation method for remaining missing values
-Execute imputation and return imputed dataframe
Arguments:
-self: the instance of the class
-data: the DataFrame to be imputed
-imputate_option: the method of imputation to be used (mean, median, mode, custom)
-custom_values: if imputate_option is custom, a dictionary of column names and values to impute with
Returns:
-an imputed DataFrame

"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: list = None,
        method: str = "mean",
        constant_val=None,
        drop_threshold: float = None,
    ):
        self.columns = columns
        self.method = method
        self.constant_val = constant_val
        self.drop_threshold = drop_threshold

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns if self.columns is not None else X.columns.tolist()
        self.columns_ = [c for c in cols if c in X.columns]

        missing_pct = X.isnull().mean() * 100
        if self.drop_threshold is not None:
            self.drop_columns_ = missing_pct[missing_pct > self.drop_threshold].index.tolist()
        else:
            self.drop_columns_ = []

        self.fill_values_ = {}
        for col in self.columns_:
            if col in self.drop_columns_:
                continue

            if self.method in ["mean", "median"]:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    continue
                fill_val = X[col].mean() if self.method == "mean" else X[col].median()
            elif self.method == "mode":
                if X[col].mode().empty:
                    continue
                fill_val = X[col].mode().iloc[0]
            elif self.method == "constant":
                fill_val = self.constant_val
            else:
                raise ValueError(f"Unsupported imputation method: {self.method}")

            if fill_val is not None and not (isinstance(fill_val, float) and np.isnan(fill_val)):
                self.fill_values_[col] = fill_val

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if getattr(self, "drop_columns_", None):
            df = df.drop(columns=[c for c in self.drop_columns_ if c in df.columns])

        for col, fill_val in getattr(self, "fill_values_", {}).items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)

        return df

    def get_missingness(self, X: pd.DataFrame) -> pd.Series:
        """
        Calculates the percentage of missing values per column.
        Returns a sorted pandas Series of only columns that have missing data.
        """
        missing_pct = X.isnull().mean() * 100
        # Filter out columns with 0% missing, and sort highest to lowest
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        return missing_pct