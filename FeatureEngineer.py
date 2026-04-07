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

class FeatureEngineer:
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()

    def binarize(self, column: str, threshold: float, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.Binarizer
        Turns a continuous numeric column into 1s and 0s based on a threshold.
        """
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return self.df
            
        target_col = new_column_name if new_column_name else f"{column}_binarized"
        
        # Returns 1 if greater than threshold, 0 otherwise
        self.df[target_col] = (self.df[column] > threshold).astype(int)
        return self.df

    def quantile_binning(self, column: str, q: int, labels: list = None, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.KBinsDiscretizer (strategy='quantile')
        Splits data into 'q' equal-sized buckets based on distribution.
        Great for handling heavy outliers without deleting them.
        """
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return self.df
            
        target_col = new_column_name if new_column_name else f"{column}_q{q}_bins"
        
        # pd.qcut automatically finds the percentiles and bins them
        self.df[target_col] = pd.qcut(self.df[column], q=q, labels=labels, duplicates='drop')
        return self.df

    def log_transform(self, column: str, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.PowerTransformer / FunctionTransformer
        Applies a log(1+x) transform to heavily skewed right-tail data.
        """
        if column not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[column]):
            return self.df
            
        target_col = new_column_name if new_column_name else f"{column}_log"
        
        # log1p is log(1 + x). It safely handles zeros so they don't turn into -infinity
        self.df[target_col] = np.log1p(self.df[column])
        return self.df

    def polynomial_interaction(self, col1: str, col2: str, new_column_name: str = None) -> pd.DataFrame:
        """
        Stolen from: sklearn.preprocessing.PolynomialFeatures
        Multiplies two continuous columns together to capture interaction effects.
        """
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return self.df
            
        target_col = new_column_name if new_column_name else f"{col1}_x_{col2}"
        self.df[target_col] = self.df[col1] * self.df[col2]
        return self.df

    def apply_custom_row_logic(self, new_column_name: str, custom_func) -> pd.DataFrame:
        """
        Your open door for domain-specific medical logic that math can't solve.
        """
        self.df[new_column_name] = self.df.apply(custom_func, axis=1)
        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        return self.df