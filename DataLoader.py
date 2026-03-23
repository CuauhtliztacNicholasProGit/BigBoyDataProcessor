"""
This Class is responsible for loading the data from a file (of various formats) into a DataFrame 
It provides methods for:
-Loading CSV, Excel, JSON, SQL, SAS, SPSS, Stata, RData, Pickles, and is amenable to expansion.

Structure:
-DataLoader class with methods for each file type
-Method to auto-detect file type and call appropriate loading method

Arguments:
-self: the instance of the class
-file_path: the path to the file to be loaded
-kwargs: additional keyword arguments to be passed to the pandas read function
Returns:
-a pandas DataFrame containing the loaded data
"""

import pandas as pd
from pathlib import Path
import pyreadr # Standard library for reading R files in Python
from sqlalchemy import create_engine # Industry standard for SQL connections

class DataLoader:
    def __init__(self, source: str, db_uri: str = None):
        """
        source: A file path OR a SQL query string.
        db_uri: Connection string required ONLY if loading via SQL.
        """
        self.source = source
        self.db_uri = db_uri
        
        # Mapping extensions (including the dot for safer parsing)
        self._supported_formats = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.json': pd.read_json,
            '.sas7bdat': pd.read_sas,
            '.sav': pd.read_spss,
            '.dta': pd.read_stata,
            '.pkl': pd.read_pickle,
            '.html': pd.read_html,
            '.parquet': pd.read_parquet,
            '.feather': pd.read_feather,
            '.rdata': self._load_rdata,
            '.rds': self._load_rdata
        }

    def _load_rdata(self, path, **kwargs) -> pd.DataFrame:
        """Helper method to handle RData files via pyreadr."""
        result = pyreadr.read_r(path, **kwargs)
        # pyreadr returns a dictionary of R objects. We grab the first DataFrame.
        return next(iter(result.values()))

    def _load_sql(self, query, **kwargs) -> pd.DataFrame:
        """Helper method to handle SQL execution."""
        if not self.db_uri:
            raise ValueError("A db_uri must be provided to load data from SQL.")
        engine = create_engine(self.db_uri)
        return pd.read_sql(query, engine, **kwargs)

    def data_loader(self, **kwargs) -> pd.DataFrame:
        """Auto-detects format and loads data. Accepts pandas kwargs."""
        
        # 1. Check if the source is a SQL query rather than a file path
        if self.source.strip().lower().startswith(('select ', 'with ')):
            return self._load_sql(self.source, **kwargs)

        # 2. File-based loading using pathlib for robust path parsing
        path_obj = Path(self.source)
        
        if not path_obj.is_file():
            raise FileNotFoundError(f"The file '{self.source}' was not found.")

        ext = path_obj.suffix.lower()
        
        if ext in self._supported_formats:
            # Execute the mapped function
            return self._supported_formats[ext](self.source, **kwargs)
        else: #
            raise ValueError(f"Unsupported file format: {ext}")