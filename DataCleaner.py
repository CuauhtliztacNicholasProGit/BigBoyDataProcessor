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

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        # Work on a copy to avoid SettingWithCopyWarning
        self.df = data.copy()
        
        # String operations
        self._str_ops = {
            'convert_to_string': lambda s: s.astype(str, errors='ignore'),
            'lowercase': lambda s: s.str.lower(),
            'strip_whitespace': lambda s: s.str.strip(),
            'strip_punctuation': lambda s: s.str.replace(f'[{re.escape(string.punctuation)}]', '', regex=True),
            'strip_underscores': lambda s: s.str.replace('_', ' ', regex=False),
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

    def clean_strings(self, columns: list, options: dict) -> pd.DataFrame:
        """
        Applies specific string cleaning operations to a targeted list of columns.
        options example: {'lowercase': True, 'custom_removal': r'\d+'}
        """
        for col in columns:
            if col not in self.df.columns:
                continue
            # Handle standard boolean options
            for opt_name, apply_opt in options.items():
                if opt_name in self._str_ops and apply_opt:
                    self.df[col] = self._str_ops[opt_name](self.df[col])
            
            # Handle custom regex removal separately since it requires a value, not a boolean
            custom_regex = options.get('custom_removal')
            if custom_regex and isinstance(custom_regex, str):
                # User passes a raw regex string of what to remove
                self.df[col] = self.df[col].apply(lambda x: re.sub(custom_regex, '', str(x)) if isinstance(x, str) else x)

                #
        return self.df

    def map_categories_regex(self, column: str, regex_mapping: dict, default_val=None, new_column_name=None) -> pd.DataFrame:
        """
        Maps a string column to categorical values using regex patterns.
        regex_mapping example: {r'(?i).*heart.*': 'Cardio', r'(?i).*trauma.*': 'ER'}
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        target_col = new_column_name if new_column_name else column
        
        # If writing to a new column, initialize it with the default value or the original data
        if target_col != column:
            self.df[target_col] = default_val if default_val is not None else self.df[column]
            
        # Apply regex replacements sequentially
        # Using pandas native replace with regex=True is highly optimized C-code under the hood
        self.df[target_col] = self.df[target_col].replace(regex_mapping, regex=True)
        
        return self.df

    def clean_numeric(self, columns: list, to_numeric: bool = False, force_dtype: str = None,  round_decimals: int = None) -> pd.DataFrame:
        """
        Safely applies numeric conversions and rounding to targeted columns.
        """
        for col in columns:
            # 1. Safely parse numbers first. 
            if col not in self.df.columns:
                continue
            if to_numeric:
                # coercion is safer now because it's only applied to columns the user explicitly selected
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            # 2. Force the specific data type. 
            if force_dtype:
                try:
                    self.df[col] = self.df[col].astype(force_dtype)
                except ValueError as e:
                    print(f"Could not force column '{col}' to dtype {force_dtype}: Error: {e}")
            
            # 3. Apply rounding if it's a valid numeric column. 
            if round_decimals is not None and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].round(round_decimals)

        return self.df
    def apply_custom_function(self, columns: list, custom_func) -> pd.DataFrame:
        """
        The 'Open Door' method. Allows you to pass any custom Python function 
        to target columns without modifying the DataCleaner class itself.
        """
        for col in columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(custom_func)
        return self.df
    
    def get_dataframe(self) -> pd.DataFrame:
        """Returns the final cleaned dataframe."""
        return self.df