"""

"""
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
    PolynomialFeatures, KBinsDiscretizer, SplineTransformer,
    FunctionTransformer, OneHotEncoder, PowerTransformer
    )
from typing import Callable, Dict, List, Optional, Union, Any
    

class ModelFormatter:
    def __init__(self, data: pd.DataFrame):
        self.df = data.copy()

    
    # text to math (one-hot encoding)
    
    def encode_categorical(self, columns: list, drop_first: bool = True) -> pd.DataFrame:
        """
        Converts text categories into binary 1s and 0s.
        drop_first=True prevents the 'Dummy Variable Trap' (perfect multicollinearity),
        which is critical for linear/logistic regression models.
        """
        existing_cols = [col for col in columns if col in self.df.columns]
        
        if not existing_cols:
            return self.df
            
        # pd.get_dummies is the safest way to encode while keeping column names readable
        self.df = pd.get_dummies(self.df, columns=existing_cols, drop_first=drop_first, dtype=int)
        return self.df

    #leveling numbers (scaling)
    def scale_numeric(self, columns: list, method: str = 'standard') -> pd.DataFrame:
        """
        Scales numeric columns so large numbers don't overpower small numbers.
        Options: 'standard' (Z-score), 'minmax' (0 to 1), 'robust' (ignores outliers)
        """
        existing_cols = [col for col in columns if col in self.df.columns]
        
        if not existing_cols:
            return self.df

        # Select the sklearn scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler() 
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            print("Invalid scaling method.")
            return self.df
            
        # Fit and transform, but map it BACK into the pandas dataframe 
        # so we don't lose our column names to a raw numpy array!
        self.df[existing_cols] = scaler.fit_transform(self.df[existing_cols])
        
        return self.df

    def get_dataframe(self) -> pd.DataFrame:
        return self.df