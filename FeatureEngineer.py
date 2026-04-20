"""
This class provides methods for creating new features from existing data. It provides methods for:
-creating interaction features (multiplying two features together)
-creating polynomial features (squaring a feature, etc)
-creating binned features (converting a continuous variable into categorical bins)
-creating date features (extracting year, month, day, etc from a datetime column)
Structure:
-FeatureEngineer class with methods for different types of feature engineering
Arguments:
-self: the instance of the class
-data: the DataFrame to be engineered
Returns:
-a DataFrame with new features added
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, operations: list = None):
        self.operations = operations

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        for op in self.operations or []:
            df = self._apply_operation(df, op)

        return df

    def binarize(self, df: pd.DataFrame, column: str, threshold: float, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.Binarizer
        Turns a continuous numeric column into 1s and 0s based on a threshold.
        """
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df
            
        target_col = new_column_name if new_column_name else f"{column}_binarized"
        
        # Returns 1 if greater than threshold, 0 otherwise
        df[target_col] = (df[column] > threshold).astype(int)
        return df

    def quantile_binning(self, df: pd.DataFrame, column: str, q: int, labels: list = None, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.KBinsDiscretizer (strategy='quantile')
        Splits data into 'q' equal-sized buckets based on distribution.
        Great for handling heavy outliers without deleting them.
        """
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df
            
        target_col = new_column_name if new_column_name else f"{column}_q{q}_bins"
        
        # pd.qcut automatically finds the percentiles and bins them
        df[target_col] = pd.qcut(df[column], q=q, labels=labels, duplicates='drop')
        return df

    def log_transform(self, df: pd.DataFrame, column: str, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.PowerTransformer / FunctionTransformer
        Applies a log(1+x) transform to heavily skewed right-tail data.
        """
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            return df
            
        target_col = new_column_name if new_column_name else f"{column}_log"
        
        # log1p is log(1 + x). It safely handles zeros so they don't turn into -infinity
        df[target_col] = np.log1p(df[column])
        return df

    def polynomial_interaction(self, df: pd.DataFrame, col1: str, col2: str, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.PolynomialFeatures
        Multiplies two continuous columns together to capture interaction effects.
        """
        if col1 not in df.columns or col2 not in df.columns:
            return df
            
        target_col = new_column_name if new_column_name else f"{col1}_x_{col2}"
        df[target_col] = df[col1] * df[col2]
        return df

    def apply_custom_row_logic(self, df: pd.DataFrame, new_column_name: str, custom_func) -> pd.DataFrame:
        """
        Your open door for domain-specific medical logic that math can't solve.
        """
        if new_column_name and custom_func:
            df[new_column_name] = df.apply(custom_func, axis=1)
        return df

    def _apply_operation(self, df: pd.DataFrame, op: dict) -> pd.DataFrame:
        op_name = op.get("op")
        if op_name == "binarize":
            return self.binarize(df, op.get("column"), op.get("threshold"), op.get("new_column_name"))
        if op_name == "quantile_binning":
            return self.quantile_binning(df, op.get("column"), op.get("q"), op.get("labels"), op.get("new_column_name"))
        if op_name == "log_transform":
            return self.log_transform(df, op.get("column"), op.get("new_column_name"))
        if op_name == "polynomial_interaction":
            return self.polynomial_interaction(df, op.get("col1"), op.get("col2"), op.get("new_column_name"))
        if op_name == "custom_row_logic":
            return self.apply_custom_row_logic(df, op.get("new_column_name"), op.get("func"))
        return df