import pandas as pd
import BioDataProcessor as bpd
#

if __name__ == "__main__":
    
    # Standard string cleaning operations
    clean_ops = {
        'lowercase': True,
        'strip_whitespace': True,
        'universal_text_scrubber': True
    }
    
    # Specific imputation strategies mapped to the logic
    imputations = [
        {'cols': ['BMI', 'Blood_Pressure'], 'method': 'median'},
        {'cols': ['Smoking_Status'], 'method': 'mode'}
    ]
    
    # Feature engineering for age acceleration factors
    engineering = [
        {'tool': 'binarize', 'column': 'Cigarettes_Per_Day', 'threshold': 0},  # fix: col→column, thresh→threshold
        {'tool': 'quantile_binning', 'column': 'Telomere_Length', 'q': 4}      # fix: col→column
    ]

    # Run the pipeline
    final_master_df = bpd.run_default_pipeline(   # fix: bpd → bpd.run_default_pipeline
        primary_file=r"C:\Users\Miste\Documents\Projects\2025-datascience-homework\STAR ICU\BigBoyDataProcessor\Phys1_Vitals_public.csv",
        output_file='cleaned_biostats_master.csv',
        
        # string_clean_cols omitted — None is the default and means all string columns
        string_clean_options=clean_ops,
        impute_drop_threshold=40.0,
        impute_strategies=imputations,             # fix: impute_strategy → impute_strategies
        engineer_steps=engineering,
        
        encode_categorical=True,
        scale_method='robust',
    )