"""
This module defines the Imputer class, which provides methods for handling missing values in a DataFrame.
It provides methods for:
-mean
-median
-mode
-custom imputation option

Structure:
-Caluculate missingness per column and print to user
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
    def __init__(self, data):
        self.data = data.copy() 
    def calculate_missingness(self):
        missingness = self.data.isnull().mean() * 100
        print("Missingness per column (%):\n", missingness)
        return missingness
    def impute_missing(self, impute_threshold = None, impute_option = None, custom_values=None):
        # ask user for missingness threshold to drop columns
        print(f"Columns with this missingness will be dropped: {impute_threshold}%")
        # 
