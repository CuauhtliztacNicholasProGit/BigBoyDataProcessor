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

class Imputer:
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()

    def get_missingness(self) -> pd.Series:
        """
        Calculates the percentage of missing values per column.
        Returns a sorted pandas Series of only columns that have missing data.
        """
        missing_pct = self.df.isnull().mean() * 100
        # Filter out columns with 0% missing, and sort highest to lowest
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        return missing_pct

    def drop_high_missingness(self, threshold: float) -> list:
        """
        Drops columns where the percentage of missing values exceeds the threshold.
        Returns the list of dropped columns so the user knows what was removed.
        """
        missing_pct = self.df.isnull().mean() * 100
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            
        return cols_to_drop

    def impute_columns(self, columns: list, method: str, constant_val=None) -> pd.DataFrame:
        """
        Fills missing values based on the chosen method.
        Safely checks if a column is numeric before applying mean/median.
        """
        for col in columns:
            if col not in self.df.columns:
                continue
                
            # Mean and Median ONLY work on numeric data
            if method in ['mean', 'median']:
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    print(f"Skipping {method} for '{col}' - it is not a numeric column.")
                    continue
                    
                fill_val = self.df[col].mean() if method == 'mean' else self.df[col].median()
                self.df[col] = self.df[col].fillna(fill_val)

            # Mode works on anything (categorical or numeric)
            elif method == 'mode':
                if not self.df[col].mode().empty:
                    fill_val = self.df[col].mode().iloc[0]
                    self.df[col] = self.df[col].fillna(fill_val)

            # Constant uses the user's provided value
            elif method == 'constant' and constant_val is not None:
                self.df[col] = self.df[col].fillna(constant_val)

        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the fully imputed dataframe."""
        return self.df