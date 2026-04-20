"""
This class provides methods for merging files together based on common kets. It provides methods for:
-merging two datasets together based on a common key column (like patient ID, sequence number, etc)

Structure:
-DataMerger class with methods for merging datasets
Arguments:
-self: the instance of the class
-data: the second DataFrame to be merged
-
Returns:
- a merged DataFrame

"""
#one hot feature explosion --- can cause sparsity
#If we have to many cats, one hot can creat problems, to sparse of a data frame
#for each column, we only have X 2% of the time or something. 
#becomes a lot harder to get meaningful pattersn
#dont want to one hot encode when lots of catagories, try courser catagories, get it down to 10 or 5
#are there natural ways to bin the data? gotta be careful cuz that needs domain experties
#text options - amkes senes or dont make sense 
# what kind of info is useful to be encoding. 
import pandas as pd

class DataMerger:
    
    @staticmethod
    def merge_datasets(
        base_df: pd.DataFrame,
        new_df: pd.DataFrame,
        on: list,
        how: str = 'left',
        filters: dict = None,
        collapse_duplicates: bool = True,
        key_scrubber_func=None,
    ) -> pd.DataFrame:
        """
        Merges datasets safely. 
        - filters: A dictionary of column:value pairs to filter the incoming dataset by.
        - collapse_duplicates: Prevents row explosion by keeping only the first instance.
        - key_scrubber_func: Optional callable to clean join keys before merge.
        """
        base_df = base_df.copy()
        new_df = new_df.copy()
        # 1. Safety Check
        for key in on:
            if key not in base_df.columns:
                print(f"ERROR: Merge key '{key}' is missing from primary dataset. Aborting.")
                return base_df
            if key not in new_df.columns:
                print(f"ERROR: Merge key '{key}' is missing from new dataset. Aborting.")
                return base_df

        # 2. Harmonizer
        print(f"Harmonizing join keys {on}...")
        for key in on:
            if key_scrubber_func:
                base_df[key] = base_df[key].apply(key_scrubber_func)
                new_df[key] = new_df[key].apply(key_scrubber_func)
            
            base_df[key] = base_df[key].astype(str).str.replace(r'\.0$', '', regex=True)
            new_df[key] = new_df[key].astype(str).str.replace(r'\.0$', '', regex=True)

        # 3. Iterative Multi-Column Filter
        if filters:
            for f_col, f_val in filters.items():
                if f_col in new_df.columns:
                    target_str = str(f_val).strip()
                    new_df[f_col] = new_df[f_col].astype(str).str.strip()
                    
                    rows_before_filter = new_df.shape[0]
                    new_df = new_df[new_df[f_col] == target_str]
                    print(f"  -> Filtered '{f_col}' == '{target_str}' (Kept {new_df.shape[0]} out of {rows_before_filter} rows).")
                else:
                    print(f"  -> WARNING: Filter column '{f_col}' not found. Skipping.")

        # 4. Deduplicator
        if collapse_duplicates:
            rows_before_dedup = new_df.shape[0]
            new_df = new_df.drop_duplicates(subset=on, keep='first')
            rows_dropped = rows_before_dedup - new_df.shape[0]
            
            if rows_dropped > 0:
                print(f"  -> Collapsed {rows_dropped} duplicate rows to protect Master Matrix.")

        # 5. Join
        merged_df = pd.merge(
            base_df, 
            new_df, 
            on=on, 
            how=how, 
            suffixes=('_x', '_y')
        )
        
        return merged_df