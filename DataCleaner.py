"""
This class is responsible for cleaning the data after it has been loaded. It provides methods for:
-cleaning up string columns:
    -converting to lower case
    -removing leading/trailing whitespace
    -removing leading/trailing punctuation
    -removing underscores
    -allow for user to specify additional characters to remove

#depending on the kind of string, might be better to 
#!!! ---make it a set of catagories that we want strings to map to. ---
 regex - built in pandas function

-----
 example of using pandas regex to map string columns to a new categorical column:
df['category'] = 0
df.loc[(df.lab.str.contains('something')==True),'category'] = 1
-----

-cleaning up numerical columns:
    -converting to numeric types (if possible)
    -rounding to specified number of decimal places

Structure:
-DataCleaner class with methods for string cleaning and numeric cleaning

Arguments: 
-self: the instance of the class
-data: the DataFrame to be cleaned
-clean_options_string: a dictionary specifying which cleaning operations to perform on string columns
-clean_options_numeric: a dictionary specifying which cleaning operations to perform on numeric columns
Returns:
-a cleaned DataFrame

"""

import pandas as pd
import numpy as np
import string
import re
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        string_columns: list = None,
        string_options: dict = None,
        numeric_columns: list = None,
        to_numeric: bool = False,
        force_dtype: str = None,
        round_decimals: int = None,
        drop_columns: list = None,
        category_mappings: list = None,
        custom_column_functions: list = None,
        custom_row_functions: list = None,
    ):
        self.string_columns = string_columns
        self.string_options = string_options
        self.numeric_columns = numeric_columns
        self.to_numeric = to_numeric
        self.force_dtype = force_dtype
        self.round_decimals = round_decimals
        self.drop_columns = drop_columns
        self.category_mappings = category_mappings
        self.custom_column_functions = custom_column_functions
        self.custom_row_functions = custom_row_functions

        # String operations
        self._str_ops = {
            'convert_to_string': lambda s: s.astype('string'),
            'lowercase': lambda s: s.astype('string').str.lower(),
            'strip_whitespace': lambda s: s.astype('string').str.strip(),
            'strip_punctuation': lambda s: s.astype('string').str.replace(f'[{re.escape(string.punctuation)}]', '', regex=True),
            'strip_underscores': lambda s: s.astype('string').str.replace('_', ' ', regex=False),
            'universal_text_scrubber': lambda s: s.apply(self.universal_text_scrubber),
        }

    @staticmethod
    def universal_text_scrubber(x):
        """
        A robust catch-all for formatting anomalies in legacy or messy data.
        """
        # 1. Preserve actual missing values
        if pd.isna(x):
            return x 
            
        # 2. Decode native byte objects immediately
        if isinstance(x, bytes): 
            x = x.decode("utf-8", errors="ignore") 
            
        x = str(x).strip()
        
        # 3. Strip stringified bytes (b'...' or b"...")
        if (x.startswith("b'") and x.endswith("'")) or (x.startswith('b"') and x.endswith('"')):
            x = x[2:-1]
            
        # 4. Destroy "Phantom NaNs" (Pandas sometimes turns empty cells into the literal word "nan")
        if x.lower() in ['nan', 'none', 'null', 'nat', '']:
            return np.nan
            
        # 5. Normalize chaotic whitespace (turns tabs, newlines, and double spaces into a single space)
        x = re.sub(r'\s+', ' ', x)
            
        return x

    def fit(self, X: pd.DataFrame, y=None):
        self.string_columns_ = self._resolve_string_columns(X)
        self.numeric_columns_ = self._resolve_numeric_columns(X)
        self.drop_columns_ = [c for c in (self.drop_columns or []) if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if getattr(self, "drop_columns_", None):
            df = self.drop_columns_from_df(df, self.drop_columns_)

        if self.string_options:
            df = self.clean_strings(df, self.string_columns_, self.string_options)

        if self.category_mappings:
            for mapping in self.category_mappings:
                df = self.map_categories_regex(
                    df,
                    column=mapping.get("column"),
                    regex_mapping=mapping.get("regex_mapping", {}),
                    default_val=mapping.get("default_val"),
                    new_column_name=mapping.get("new_column_name"),
                )

        if self.numeric_columns_:
            df = self.clean_numeric(
                df,
                self.numeric_columns_,
                to_numeric=self.to_numeric,
                force_dtype=self.force_dtype,
                round_decimals=self.round_decimals,
            )

        if self.custom_column_functions:
            for entry in self.custom_column_functions:
                df = self.apply_custom_function(df, entry.get("columns", []), entry.get("func"))

        if self.custom_row_functions:
            for entry in self.custom_row_functions:
                df = self.apply_custom_row_logic(df, entry.get("new_column_name"), entry.get("func"))

        return df

    def clean_strings(self, df: pd.DataFrame, columns: list, options: dict) -> pd.DataFrame:
        """
        Applies specific string cleaning operations to a targeted list of columns.
        options example: {'lowercase': True, 'custom_removal': r'\d+'}
        """
        for col in columns:
            if col not in df.columns:
                continue
            # Handle standard boolean options
            for opt_name, apply_opt in options.items():
                if opt_name in self._str_ops and apply_opt:
                    df[col] = self._str_ops[opt_name](df[col])
            
            # Handle custom regex removal separately since it requires a value, not a boolean
            custom_regex = options.get('custom_removal')
            if custom_regex and isinstance(custom_regex, str):
                # User passes a raw regex string of what to remove
                df[col] = df[col].astype('string').str.replace(custom_regex, '', regex=True)

        return df

    def map_categories_regex(self, df: pd.DataFrame, column: str, regex_mapping: dict, default_val=None, new_column_name=None) -> pd.DataFrame:
        """
        Maps a string column to categorical values using regex patterns.
        regex_mapping example: {r'(?i).*heart.*': 'Cardio', r'(?i).*trauma.*': 'ER'}
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        target_col = new_column_name if new_column_name else column
        
        # If writing to a new column, initialize it with the default value or the original data
        if target_col != column:
            df[target_col] = default_val if default_val is not None else df[column]
            
        # Apply regex replacements sequentially
        # Using pandas native replace with regex=True is highly optimized C-code under the hood
        df[target_col] = df[target_col].replace(regex_mapping, regex=True)

        return df

    def clean_numeric(self, df: pd.DataFrame, columns: list, to_numeric: bool = False, force_dtype: str = None,  round_decimals: int = None) -> pd.DataFrame:
        """
        Safely applies numeric conversions and rounding to targeted columns.
        """
        for col in columns:
            # 1. Safely parse numbers first. 
            if col not in df.columns:
                continue
            if to_numeric:
                # coercion is safer now because it's only applied to columns the user explicitly selected
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 2. Force the specific data type. 
            if force_dtype:
                try:
                    df[col] = df[col].astype(force_dtype)
                except ValueError as e:
                    print(f"Could not force column '{col}' to dtype {force_dtype}: Error: {e}")
            
            # 3. Apply rounding if it's a valid numeric column. 
            if round_decimals is not None and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(round_decimals)

        return df

    def drop_columns_from_df(self, df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
        """
        Safely removes a list of columns from the dataframe.
        """
        # Find which columns actually exist in the dataframe
        existing_cols = [c for c in columns_to_drop if c in df.columns]
        missing_cols = [c for c in columns_to_drop if c not in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            print(f"  -> Successfully dropped {len(existing_cols)} columns: {', '.join(existing_cols)}")
            
        if missing_cols:
            print(f"  -> WARNING: Could not find these columns to drop: {', '.join(missing_cols)}")
            
        return df

    def apply_custom_function(self, df: pd.DataFrame, columns: list, custom_func) -> pd.DataFrame:
        """
        The 'Open Door' method. Allows you to pass any custom Python function 
        to target columns without modifying the DataCleaner class itself.
        """
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(custom_func)
        return df

    def apply_custom_row_logic(self, df: pd.DataFrame, new_column_name: str, custom_func) -> pd.DataFrame:
        if new_column_name and custom_func:
            df[new_column_name] = df.apply(custom_func, axis=1)
        return df

    def _resolve_string_columns(self, X: pd.DataFrame) -> list:
        if self.string_columns is None:
            return X.select_dtypes(include=['object', 'category']).columns.tolist()
        return [c for c in self.string_columns if c in X.columns]

    def _resolve_numeric_columns(self, X: pd.DataFrame) -> list:
        if self.numeric_columns is None:
            return X.select_dtypes(include=['number']).columns.tolist()
        return [c for c in self.numeric_columns if c in X.columns]