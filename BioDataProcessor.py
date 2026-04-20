"""
BioDataProcessor.py  —  v3.1
Single-entry-point data processing pipeline for ML and predictive tasks.

Overview
--------
BioDataProcessor provides a fluent, stage-based workflow for turning raw
clinical data into a modeling-ready pandas DataFrame. Each stage is a pure
function that accepts a DataFrame first and returns a DataFrame, making the
pipeline easy to debug, reorder, and chain with pandas .pipe().

Default stage order
-------------------
1. load_data           Read the primary dataset from disk.
2. clean_text          Normalize strings and join keys.
3. merge_datasets      Left-join supplemental datasets safely.
4. validate_merge      Resolve _x and _y column collisions.
5. deduplicate         Remove duplicate rows with explicit keep logic.
6. normalize_lab_units Convert measurements to a canonical unit.
7. impute_missing      Drop sparse columns and fill remaining nulls.
8. engineer_features   Create transformed or derived features.
9. format_for_ml       One-hot encode categoricals and scale numerics.
10. save_output        Save the processed result to CSV.

Recommended usage patterns
--------------------------
1. Use process_data(...) for a complete end-to-end run.
2. Use process_data_from_config(config) when you want reproducible runs.
3. Use the individual stage functions with pandas .pipe() when experimenting.

Minimal example
---------------
processed_df = process_data(
    primary_file='my_input.csv',
    output_file='processed_output.csv'
)

Config-driven example
---------------------
config = PIPELINE_PARAMETER_TEMPLATE.copy()
config['primary_file'] = 'my_input.csv'
config['output_file'] = 'processed_output.csv'
processed_df = process_data_from_config(config)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from DirectoryCrawler import DirectoryCrawler
from DataLoader    import DataLoader
from DataMerger    import DataMerger
from DataCleaner   import DataCleaner
from Imputer       import Imputer
from FeatureEngineer import FeatureEngineer
from HyperParameterTuner import HyperparameterTuner


PIPELINE_PARAMETER_TEMPLATE = {
    # Stage 0: auto_prepare_merges
    'dir_path': 'path/to/directory', 
    'file_extension': '.csv',
    'name_pattern': None,  #
    'on_keys': ['SEQN'],   # join keys for the merges
    'how': 'left',         # merge method for all merges (default: left join
    # Stage 1: load
    'primary_file': 'path/to/primary_dataset.csv',
    'output_file': 'processed_output.csv',
    'load_kwargs': {},

    # Stage 2: text cleaning
    'clean_columns': None,
    'clean_options': {
        'universal_text_scrubber': True,
        'lowercase': True,
        'strip_whitespace': True,
    },
    'drop_columns': [],

    # Stage 3: merges
    'merge_configs': [
        # {
        #     'file': 'path/to/secondary_file.csv',
        #     'on': ['PATIENT_ID'],
        #     'how': 'left',
        #     'filters': {},
        #     'collapse_duplicates': True,
        # }
    ],

    # Stage 5: deduplication
    'dedupe_subset': None,
    'dedupe_keep': 'first',
    'dedupe_sort_by': None,
    'dedupe_sort_ascending': True,

    # Stage 6: lab normalization
    'lab_configs': [],

    # Stage 7: imputation
    'drop_threshold': 50.0,
    'impute_strategies': [
        # {'cols': ['AGE', 'BMI'], 'method': 'median'},
        # {'cols': ['SITE'], 'method': 'mode'},
        # {'cols': ['NOTES'], 'method': 'constant', 'constant_val': 'unknown'},
    ],

    # Stage 8: feature engineering
    'feature_steps': [
        # {'tool': 'log_transform', 'column': 'LOS_DAYS'},
        # {'tool': 'binarize', 'column': 'RISK_SCORE', 'threshold': 0.5},
        # {'tool': 'polynomial_interaction', 'col1': 'AGE', 'col2': 'BMI'},
    ],

    # Stage 9: ML formatting
    'encode_categorical': True,
    'exclude_encode_prefixes': ('ID', 'LNK_', 'SITE'),
    'scale_method': 'standard',
    'exclude_scale_cols': [],
    'default_scale_exclusions': ['ID', 'SEQNO'],
}


# STAGE 0 - Crawl directory to build merge_configs (optional for large foulder sets)
def directory_crawler(
    dir_path: str, 
    file_extension: str, 
    name_pattern: str = None
) -> list:
    """
    Utility function to crawl a working directory for files matching specific criteria,
    returning a list of file paths that can be used in the merge stage.
    """
    crawler = DirectoryCrawler(dir_path)
    file_paths = crawler.find_files(extension=file_extension, name_pattern=name_pattern)

    return(file_paths)

#  STAGE 1 — Load

def load_data(
    primary_file: str,
    loader_cls=DataLoader,
    **kwargs
) -> pd.DataFrame:
    """Load the primary dataset from any supported file format."""
    print(_header("Loading primary dataset"))
    df = loader_cls(primary_file).data_loader(**kwargs)
    print(f"  Loaded {df.shape[0]:,} rows × {df.shape[1]} columns from '{primary_file}'")
    return df


#  STAGE 2 — Clean text  (run BEFORE merging)

def clean_text(
    df: pd.DataFrame,
    columns: list = None,          # None → all object/category columns
    options: dict = None,          # cleaning flags; see defaults below
    drop_columns: list = None,
    cleaner_cls=DataCleaner
) -> pd.DataFrame:
    """
    Scrub string columns.  Should be called on EVERY dataset before any merge
    so that join keys are in a consistent format.

    Default options apply the universal_text_scrubber plus lowercase + strip.
    Pass options={} to skip all cleaning while still allowing column drops.
    """
    default_options = {
        'universal_text_scrubber': True,
        'lowercase': True,
        'strip_whitespace': True,
    }
    opts = options if options is not None else default_options

    print(_header("Cleaning text columns"))
    cleaner = cleaner_cls(
        string_columns=columns,
        string_options=opts,
        drop_columns=drop_columns,
    )

    cleaner.fit(df)

    if opts:
        target_cols = getattr(cleaner, "string_columns_", [])
        if target_cols:
            print(f"  Cleaned {len(target_cols)} string column(s).")

    return cleaner.transform(df)


#  STAGE 3 — Merge

def merge_datasets(
    df: pd.DataFrame,
    merge_configs: list = None,    # [{file, on, how, filters, collapse_duplicates}]
    loader_cls=DataLoader,
    merger_cls=DataMerger,
    cleaner_cls=DataCleaner
) -> pd.DataFrame:
    """
    Left-join one or more supplemental datasets onto the primary dataframe.

    Each config dict supports:
        file               : path to the file to merge in
        on                 : list of join-key column names
        how                : 'left' (default), 'inner', 'outer', 'right'
        filters            : {col: value} dict to pre-filter the incoming file
        collapse_duplicates: True (default) — deduplicate incoming file on
                             join keys BEFORE merging to prevent row explosion

    Text cleaning is applied to every incoming dataset before the join so
    join keys are always in the same normalised format.
    """
    if not merge_configs:
        return df

    print(_header("Merging supplemental datasets"))
    for i, cfg in enumerate(merge_configs, 1):
        fpath  = cfg['file']
        on     = cfg.get('on', [])
        how    = cfg.get('how', 'left')

        print(f"  [{i}] Merging '{fpath}' on {on} ({how} join)...")

        new_df = loader_cls(fpath).data_loader()

        # Clean the incoming dataset's text (especially the join keys) first
        new_df = clean_text(new_df, cleaner_cls=cleaner_cls)

        df = merger_cls.merge_datasets(
            base_df=df,
            new_df=new_df,
            on=on,
            how=how,
            filters=cfg.get('filters', {}),
            collapse_duplicates=cfg.get('collapse_duplicates', True),
            key_scrubber_func=cleaner_cls.universal_text_scrubber
        )

        # Immediately validate — fail loud rather than silently corrupt
        df = validate_merge(df, on=on)

    return df
# STAGE 3.5 - Auto-prepare merge_configs from a directory
def auto_prepare_merges(
        folder_path, 
        extension, 
        on_keys, 
        how = 'left'
    ):
    """
    Scans a folder and automatically creates the 'merge_configs' list.
    No manual typing of filenames required.
    """
    from DirectoryCrawler import DirectoryCrawler
    
    # Scout finds the files
    crawler = DirectoryCrawler(folder_path)
    file_paths = crawler.find_files(extension=extension)
    
    # Build the instruction list for the loop in BioDataProcessor
    return [{'file': p, 'on': on_keys, 'how': how, 'collapse_duplicates': True} for p in file_paths]


#  STAGE 4 — Validate merge (no _x / _y leaks)

def validate_merge(
    df: pd.DataFrame,
    on: list = None,
    collision_strategy: str = 'keep_base'
) -> pd.DataFrame:
    """
    Detects columns that ended up with _x / _y suffixes after a merge,
    which means the same column name existed in both datasets and pandas
    couldn't reconcile them automatically.

    collision_strategy options
    --------------------------
    'keep_base'  (default) — keep the _x (base/primary) version, drop _y
    'keep_new'             — keep the _y (incoming) version, drop _x
    'keep_both'            — rename to <col>_base and <col>_new and keep both
                             (useful when both versions carry real information)

    After resolution the clean column is restored to its original name.
    """
    collision_bases = set()
    for col in df.columns:
        if col.endswith('_x') or col.endswith('_y'):
            collision_bases.add(col[:-2])

    if not collision_bases:
        return df

    print(f"\n  ⚠  Merge collision detected in: {sorted(collision_bases)}")
    print(f"     Strategy: '{collision_strategy}'")

    for base in sorted(collision_bases):
        x_col = f"{base}_x"
        y_col = f"{base}_y"

        if collision_strategy == 'keep_base':
            df = df.drop(columns=[y_col], errors='ignore')
            df = df.rename(columns={x_col: base})
            print(f"     '{base}': kept base (_x), dropped incoming (_y).")

        elif collision_strategy == 'keep_new':
            df = df.drop(columns=[x_col], errors='ignore')
            df = df.rename(columns={y_col: base})
            print(f"     '{base}': kept incoming (_y), dropped base (_x).")

        elif collision_strategy == 'keep_both':
            df = df.rename(columns={x_col: f"{base}_base", y_col: f"{base}_new"})
            print(f"     '{base}': kept both as '{base}_base' / '{base}_new'.")

        else:
            raise ValueError(
                f"Unknown collision_strategy '{collision_strategy}'. "
                "Choose 'keep_base', 'keep_new', or 'keep_both'."
            )

    return df

#  STAGE 5 — Deduplicate

def deduplicate(
    df: pd.DataFrame,
    subset: list = None,           # Columns to consider; None = all columns
    keep: str = 'first',           # 'first', 'last', or False (drop ALL dupes)
    sort_by: list = None,          # Sort before deduplicating so 'first'/'last'
    sort_ascending: bool = True    # is deterministic and meaningful
) -> pd.DataFrame:
    """
    Remove duplicate rows with explicit, documented logic.

    keep='first' after sort_by means: among duplicates, retain the row that
    sorts earliest.  For longitudinal data (e.g. repeated visits) you would
    typically sort_by=['PATIENT_ID', 'VISIT_DATE'] and keep='last' to retain
    the most recent record per patient.

    Examples
    --------
    # Keep the most recent lab result per patient
    deduplicate(df, subset=['PATIENT_ID', 'LAB_CODE'],
                sort_by=['PATIENT_ID', 'RESULT_DATE'], keep='last')

    # Keep first occurrence of each ID (enrolment record)
    deduplicate(df, subset=['ID'], keep='first')
    """
    rows_before = len(df)

    if sort_by:
        df = df.sort_values(by=sort_by, ascending=sort_ascending)

    df = df.drop_duplicates(subset=subset, keep=keep)
    dropped = rows_before - len(df)

    if dropped:
        print(f"  Deduplication: removed {dropped:,} rows "
              f"(keep='{keep}', subset={subset}).")
    else:
        print("  Deduplication: no duplicate rows found.")

    return df.reset_index(drop=True)


#  STAGE 6 — Lab unit normalisation
def normalize_lab_units(
    df: pd.DataFrame,
    lab_configs: list = None
) -> pd.DataFrame:
    """
    Ensures every lab value is expressed in a single, consistent unit.

    Why this matters
    ----------------
    If glucose is recorded in both mg/dL and mmol/L in the same column,
    a model will treat 5.5 (mmol/L) and 99 (mg/dL) as completely different
    values even though they are clinically identical.  This stage detects
    mixed-unit columns and converts everything to a chosen canonical unit.

    lab_configs  — list of dicts, one per lab test that needs normalisation:

        {
          'value_col'    : 'GLUCOSE',         # numeric column with the measurement
          'unit_col'     : 'GLUCOSE_UNIT',    # categorical column holding the unit label
          'canonical'    : 'mg/dL',           # unit every value will be converted TO
          'conversions'  : {                  # conversion factors FROM each other unit
              'mmol/L': 6.945,               # value_in_canonical = raw * factor
              'µmol/L': 0.006945,
          }
        }

    If lab_configs is None or empty the stage is skipped (safe default).
    """
    if not lab_configs:
        return df

    print(_header("Normalising lab units"))

    for cfg in lab_configs:
        val_col    = cfg['value_col']
        unit_col   = cfg.get('unit_col')
        canonical  = cfg['canonical']
        conversions = cfg.get('conversions', {})

        if val_col not in df.columns:
            print(f"  WARNING: value column '{val_col}' not found. Skipping.")
            continue

        if unit_col and unit_col not in df.columns:
            print(f"  WARNING: unit column '{unit_col}' not found. Skipping.")
            continue

        if not unit_col:
            # No unit column — nothing to reconcile
            print(f"  '{val_col}': no unit column provided, skipping.")
            continue

        unique_units = df[unit_col].dropna().unique()
        non_canonical = [u for u in unique_units if str(u).strip() != canonical]

        if not non_canonical:
            print(f"  '{val_col}': all values already in '{canonical}'. No action needed.")
            continue

        print(f"  '{val_col}': converting {non_canonical} → '{canonical}'")

        for unit in non_canonical:
            if unit not in conversions:
                print(f"    WARNING: no conversion factor for unit '{unit}' in '{val_col}'. "
                      "Those rows will be left unconverted — add a factor to lab_configs.")
                continue

            factor = conversions[unit]
            mask = df[unit_col].astype(str).str.strip() == str(unit)
            df.loc[mask, val_col] = df.loc[mask, val_col] * factor
            df.loc[mask, unit_col] = canonical
            print(f"    Converted {mask.sum():,} rows from '{unit}' (×{factor}).")

    return df

#  STAGE 7 — Impute

def impute_missing(
    df: pd.DataFrame,
    drop_threshold: float = 50.0,  # Drop columns with > X% missing
    strategies: list = None,       # [{cols: [...], method: 'median'}]
    imputer_cls=Imputer
) -> pd.DataFrame:
    """
    Handle missing values in two phases:
      1. Drop columns that are too sparse to be useful.
      2. Fill remaining NaNs using per-column strategies.

    strategies example
    ------------------
    [
        {'cols': ['AGE', 'BMI'],   'method': 'median'},
        {'cols': ['SITE'],         'method': 'mode'},
        {'cols': ['SCORE_NOTES'],  'method': 'constant', 'constant_val': 'unknown'},
    ]
    """
    print(_header("Imputing missing values"))
    df_out = df

    if drop_threshold is not None:
        dropper = imputer_cls(
            drop_threshold=drop_threshold,
            method="constant",
            constant_val=None,
        )
        df_out = dropper.fit_transform(df_out)
        dropped = getattr(dropper, "drop_columns_", [])
        if dropped:
            print(f"  Dropped {len(dropped)} column(s) exceeding {drop_threshold}% missingness: "
                  f"{dropped}")

    if strategies:
        for strat in strategies:
            imputer = imputer_cls(
                columns=strat.get('cols', []),
                method=strat.get('method', 'mean'),
                constant_val=strat.get('constant_val', None),
            )
            df_out = imputer.fit_transform(df_out)

    return df_out

#  STAGE 8 — Feature engineering

def engineer_features(
    df: pd.DataFrame,
    steps: list = None,
    engineer_cls=FeatureEngineer
) -> pd.DataFrame:
    """
    Create derived features.

    steps is a list of dicts.  The 'tool' key selects the method; all other
    keys are passed as keyword arguments to that method.

    Available tools
    ---------------
    binarize              col, threshold, [new_column_name]
    quantile_binning      col, q, [labels], [new_column_name]
    log_transform         col, [new_column_name]
    polynomial_interaction col1, col2, [new_column_name]
    custom_logic          new_column_name, custom_func

    Example
    -------
    steps=[
        {'tool': 'log_transform',   'column': 'LOS_DAYS'},
        {'tool': 'binarize',        'column': 'RISK_SCORE', 'threshold': 0.5},
        {'tool': 'polynomial_interaction', 'col1': 'AGE', 'col2': 'BMI'},
    ]
    """
    if not steps:
        return df

    print(_header("Engineering features"))

    operations = []
    for step in steps:
        params = step.copy()
        tool_name = params.pop('tool', None)
        if tool_name == 'custom_logic':
            tool_name = 'custom_row_logic'

        if tool_name:
            params['op'] = tool_name
            operations.append(params)
            print(f"  Engineered feature using '{tool_name}'.")
        else:
            print("  WARNING: Missing tool name in step. Skipping.")

    engineer = engineer_cls(operations=operations)
    return engineer.fit_transform(df)


#  STAGE 9 — Format for ML

def format_for_ml(
    df: pd.DataFrame,
    encode_categorical: bool = True,
    exclude_encode_prefixes: tuple = ('ID', 'LNK_', 'SITE'),
    scale_method: str = 'standard',        # 'standard', 'minmax', 'robust', or None
    exclude_scale_cols: list = None,       # Columns to protect from scaling (e.g. sequence keys)
    default_scale_exclusions: list = None
) -> pd.DataFrame:
    """
    Final ML-readiness pass:
      • One-hot encode categorical columns (skip ID-like columns).
      • Scale numeric columns (skip explicit exclusions).

    Pure-function, pipe-friendly implementation.
    """
    print(_header("Formatting for ML"))
    df_out = df.copy()

    if encode_categorical:
        cat_cols = df_out.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols
                    if not str(c).upper().startswith(tuple(p.upper() for p in exclude_encode_prefixes))]
        if cat_cols:
            df_out = pd.get_dummies(df_out, columns=cat_cols, drop_first=True, dtype=int)
            print(f"  One-hot encoded {len(cat_cols)} column(s).")

    if scale_method:
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
        }
        scaler = scaler_map.get(scale_method)
        if scaler is None:
            print(f"  WARNING: Unknown scaling method '{scale_method}'. Skipping scaling.")
            return df_out

        base_exclusions = default_scale_exclusions if default_scale_exclusions is not None else ['ID', 'SEQNO']
        protected = set(exclude_scale_cols or []) | set(base_exclusions)
        num_cols = [
            c for c in df_out.select_dtypes(include=['int64', 'float64']).columns
            if c not in protected
        ]
        if num_cols:
            df_out.loc[:, num_cols] = scaler.fit_transform(df_out[num_cols])
            print(f"  Scaled {len(num_cols)} column(s) using '{scale_method}'.")

    return df_out

# STAGE 9.5 - Hyperparameter tunning. 
def hyper_parameter_tuner(
    model,
    param_space: dict,
    X,
    y,
    method: str = 'random',   # matches HyperparameterTuner default
    cv: int = 5,
    n_iter: int = 10,
    scoring=None,
    n_jobs: int = 1,
    random_state: int = 42
):
    tuner = HyperparameterTuner(
        model=model,
        param_space=param_space,
        method=method,
        cv=cv,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state
    )
    return tuner.fit(X, y)  # X_train and y_train should be defined in the context where this is called


def process_data(
    primary_file: str,
    output_file: str = None,
    load_kwargs: dict = None,
    clean_columns: list = None,
    clean_options: dict = None,
    drop_columns: list = None,
    merge_configs: list = None,
    dedupe_subset: list = None,
    dedupe_keep: str = 'first',
    dedupe_sort_by: list = None,
    dedupe_sort_ascending: bool = True,
    lab_configs: list = None,
    drop_threshold: float = 50.0,
    impute_strategies: list = None,
    feature_steps: list = None,
    encode_categorical: bool = True,
    exclude_encode_prefixes: tuple = ('ID', 'LNK_', 'SITE'),
    scale_method: str = 'standard',
    exclude_scale_cols: list = None,
    default_scale_exclusions: list = None,
) -> pd.DataFrame:
    """
    Execute the full BioDataProcessor pipeline.

    Parameters
    ----------
    primary_file : str
        Path to the main input dataset.
    output_file : str, optional
        CSV path for saving the final processed DataFrame.
    load_kwargs : dict, optional
        Extra keyword arguments passed to the file loader.
    clean_columns : list, optional
        Subset of text columns to clean. If None, all object and category columns are used.
    clean_options : dict, optional
        Text cleaning flags such as lowercase, strip_whitespace, or universal_text_scrubber.
    drop_columns : list, optional
        Columns to remove early in the pipeline.
    merge_configs : list, optional
        List of merge instructions. Each item should contain file, on, how, filters,
        and collapse_duplicates.
    dedupe_subset : list, optional
        Columns used to identify duplicates.
    dedupe_keep : str, optional
        Which duplicate to keep: first, last, or False.
    dedupe_sort_by : list, optional
        Columns to sort by before deduplication.
    dedupe_sort_ascending : bool, optional
        Sort direction used before deduplication.
    lab_configs : list, optional
        Unit-normalization rules for lab values.
    drop_threshold : float, optional
        Drop columns whose percent missing exceeds this threshold.
    impute_strategies : list, optional
        Per-column fill strategies using mean, median, mode, or constant.
    feature_steps : list, optional
        Feature-engineering instructions such as log transforms or interactions.
    encode_categorical : bool, optional
        Whether to one-hot encode categorical features.
    exclude_encode_prefixes : tuple, optional
        Prefixes that identify categorical columns to exclude from one-hot encoding.
    scale_method : str, optional
        Numeric scaling strategy: standard, minmax, robust, or None.
    exclude_scale_cols : list, optional
        Numeric columns to protect from scaling.
    default_scale_exclusions : list, optional
        Baseline protected columns, usually identifiers.

    Returns
    -------
    pd.DataFrame
        The fully processed DataFrame.
    """
    df = (
        load_data(primary_file, **(load_kwargs or {}))
        .pipe(clean_text, columns=clean_columns, options=clean_options, drop_columns=drop_columns)
        .pipe(merge_datasets, merge_configs=merge_configs)
        .pipe(deduplicate, subset=dedupe_subset, keep=dedupe_keep,
              sort_by=dedupe_sort_by, sort_ascending=dedupe_sort_ascending)
        .pipe(normalize_lab_units, lab_configs=lab_configs)
        .pipe(impute_missing, drop_threshold=drop_threshold, strategies=impute_strategies)
        .pipe(engineer_features, steps=feature_steps)
        .pipe(
            format_for_ml,
            encode_categorical=encode_categorical,
            exclude_encode_prefixes=exclude_encode_prefixes,
            scale_method=scale_method,
            exclude_scale_cols=exclude_scale_cols,
            default_scale_exclusions=default_scale_exclusions,
        )
    )

    if output_file:
        return save_output(df, output_file)
    return df


def process_data_from_config(config: dict) -> pd.DataFrame:
    """
    Run the full pipeline from a single parameter dictionary.

    This helper is intended for reproducible experiments and scripted runs.
    It validates that the required primary_file key exists and then forwards
    the configuration into process_data.
    """
    if not isinstance(config, dict):
        raise TypeError('config must be a dictionary of process_data parameters.')

    if not config.get('primary_file'):
        raise ValueError("config must include a non-empty 'primary_file' entry.")

    valid_keys = set(PIPELINE_PARAMETER_TEMPLATE.keys())
    unknown_keys = sorted(set(config.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f'Unknown config key(s): {unknown_keys}')

    return process_data(**config)

#  STAGE 10 — Save

def save_output(
    df: pd.DataFrame,
    output_file: str = 'processed_output.csv'   # safe default — never crashes
) -> pd.DataFrame:
    """Write the final DataFrame to CSV."""
    if not output_file.endswith('.csv'):
        output_file += '.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'='*40}\nPROCESSING COMPLETE\nShape: {df.shape} | Saved to '{output_file}'\n{'='*40}\n")
    return df


if __name__ == '__main__':
    # Example 1: full pipeline from a parameter dictionary
    # config = PIPELINE_PARAMETER_TEMPLATE.copy()
    # config['primary_file'] = 'input.csv'
    # config['output_file'] = 'processed_output.csv'
    # processed_df = process_data_from_config(config)

    # Example 2: ad hoc fluent chaining with .pipe()
    # processed_df = (
    #     load_data('input.csv')
    #     .pipe(clean_text, options={'universal_text_scrubber': True, 'lowercase': True, 'strip_whitespace': True})
    #     .pipe(impute_missing, drop_threshold=50.0, strategies=[{'cols': ['AGE', 'BMI'], 'method': 'median'}])
    #     .pipe(engineer_features, steps=[{'tool': 'log_transform', 'column': 'LOS_DAYS'}])
    #     .pipe(format_for_ml, scale_method='standard')
    # )
    pass


def _header(title: str) -> str:
    return f"\n{'='*40}\n{title}\n{'='*40}"