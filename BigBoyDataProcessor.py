'''
This is a data processing program for Machine Learning and other predictive tasks.

Data processing provides the following functionality:
-loading data from various file formats (CSV, Excel, JSON, SQL, SAS, SPSS, Stata, RData, Pickles, etc)
-cleaning up string columns (converting to lowercase, etc)
-filling in missing values 
-converting categorical columns to a series of binary columns

Data prep provides the following functionality:
-feature engineering
-combining multiple datasets into a single dataset that is ready to be passed into a prediction model

Manual workflow outline(reference 02.24.2026 meeting)

'''
import pandas as pd
from DataLoader import DataLoader
from DataCleaner import DataCleaner
#from imputer import Imputer
#from feature_engineering import FeatureEngineer
#from data_merger import DataMerger

def main():
    # Step 1: Load data
    file1 = input("Enter path to first dataset: ")
    loader = DataLoader(file1)
    df = loader.data_loader() # this will auto-detect file type and load accordingly
    print("Data loaded!", df.head(20))
    
    # Step 2: Clean strings (user can specify columns)
    
    cleaner = DataCleaner(df)
    # For simplicity, auto‑detect object columns
    str_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Available string columns:\n{str_cols}")

    while True:
        #ask for columns to target
        cols_input = input("Enter columns to clean (comma-separated), type 'global_strip' to select all string columns or leave blank to skip: ")
        if cols_input.lower() in ['', ' ', 'none']:
            print("Skipping cleaning.")
            break
        cols_to_clean = [col.strip() for col in cols_input.split(',')]
        valid_cols = [col for col in cols_to_clean if col in df.columns]

        if not valid_cols:
            print("No valid columns entered. Please try again.")
            continue
        #display cleaning menu
        print(f"---Cleaning Menu for {valid_cols}---\n")
        print("0: Convert to string (if not already)")
        print("1: Lowercase")
        print("2: Strip whitespace")
        print("3: Strip special characters")
        print("4: Strip underscores")
        print("5: Custom regex removal")
        print("6: Universal Text Scrubber (handles messy legacy data)")

        ops_input = input("Enter cleaning options to apply operations (comma-separated, e.g. 1,2,5): ").strip()
        ops_list = [op.strip() for op in ops_input.split(',')]

        #Build options dict
        options = {
            'convert_to_string': '0' in ops_list, # default to False unless user selects it
            'lowercase': '1' in ops_list,
            'strip_whitespace': '2' in ops_list,
            'strip_punctuation': '3' in ops_list,
            'strip_underscores': '4' in ops_list,
            "custom_removal": '5' in ops_list, # if user selects universal text scrubber, we will prompt them for a custom regex pattern to remove
            'universal_text_scrubber': '6' in ops_list
        }
        if '5' in ops_list:
            custom_regex = input("Enter custom regex pattern to remove: ")
            options['custom_removal'] = custom_regex
        cleaner.clean_strings(valid_cols, options)
        print(f"Cleaning applied to columns: {valid_cols}")
        print(cleaner.df[valid_cols].head(20)) # show cleaned columns

        # ask if they want to clean more columns
        more_clean = input("Clean more columns? (y/n): ").strip().lower()
        if more_clean != 'y':
            break
        #loop back if input is invalid
    df = cleaner.df # update df with cleaned version
    print #

        
    """
    # Step 3: Handle missing values (user chooses method)
    imputer = Imputer(df)
    print("Missing values per column:\n", df.isnull().sum())
    method = input("Choose imputation method (mean/median/most_frequent/constant): ").strip().lower()
    if method in ['mean','median','most_frequent','constant']:
        df = imputer.impute_missing(method=method)
    else:
        print("Invalid method, skipping imputation.")
    print("Missing values after imputation:\n", df.isnull().sum())

    # Step 4: Convert categorical columns to dummies and one hot encoding
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    if cat_cols:
        df = cleaner.categorical_to_dummies(cat_cols)
        print("Categorical columns converted.")

    # Step 5: Feature engineering (example)
    engineer = FeatureEngineer(df)
    # Ask user if they want to add features...
    # (You can add loops to let user define new features)

    # Step 6: Merge with another dataset if desired (maybe for BEFORE processing, or AFTER processing)
    #consider merger IDs
    merge_another = input("Merge with another dataset? (y/n): ").lower()
    if merge_another == 'y':
        file2 = input("Enter path to second dataset: ")
        loader2 = DataLoader(file2)
        df2 = loader2.load_csv()
        merge_key = input("Enter column name to merge on: ")
        df = DataMerger.merge_datasets(df, df2, on=merge_key)
        print("Datasets merged. New shape:", df.shape)

    # Final dataset ready for modeling
    df.to_csv("processed_data.csv", index=False)
    print("Processing complete. Output saved to processed_data.csv")
    """

if __name__ == "__main__":
    main()
    