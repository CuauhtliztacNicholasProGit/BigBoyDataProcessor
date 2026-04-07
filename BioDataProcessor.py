"""
BioDataProcessor.py  —  v3.0
Single-entry-point data processing pipeline for ML and predictive tasks.

Design philosophy
-----------------
• ONE function you call without thinking: `run_default_pipeline()`
• Every stage is a separate, named function so you can call any subset
  in any order when you need a custom sequence.
• All defaults live here — no manual inputs required for a standard run.
• Flexible: override any default by passing kwargs to the stage functions
  directly.

Stage order (default pipeline)
-------------------------------
1. load          – read primary file into a DataFrame
2. clean_text    – scrub string columns BEFORE any merge
3. merge         – left-join additional datasets one at a time
4. validate_merge– detect and resolve accidental _x/_y column collisions
5. deduplicate   – drop duplicates with explicit keep logic
6. normalize_labs– ensure lab values share a single unit per test
7. impute        – drop high-missingness columns, then fill remaining NaNs
8. engineer      – create derived / transformed features
9. format        – one-hot encode categoricals, scale numerics
10. save         – write final CSV

Bugs fixed vs v2.0
------------------
- imputer.get_dataframe was missing () — assigned the method, not the result
- FeatureEngineer methods referenced `self.df[col]` (undefined) instead of
  `self.df[column]`
- output_file.endswith() crashed when output_file was None
- merge _x/_y collisions were silently passed downstream
"""

import pandas as pd
import numpy as np

from DataLoader    import DataLoader
from DataMerger    import DataMerger
from DataCleaner   import DataCleaner
from Imputer       import Imputer
from FeatureEngineer import FeatureEngineer
from ModelFormater import ModelFormatter



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
    cleaner = cleaner_cls(df)

    if drop_columns:
        cleaner.drop_columns(drop_columns)

    if opts:
        target_cols = (
            cleaner.df.select_dtypes(include=['object', 'category']).columns.tolist()
            if columns is None
            else [c for c in columns if c in cleaner.df.columns]
        )
        if target_cols:
            cleaner.clean_strings(target_cols, opts)
            print(f"  Cleaned {len(target_cols)} string column(s).")

    return cleaner.get_dataframe()


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
            collapse_duplicates=cfg.get('collapse_duplicates', True)
        )

        # Immediately validate — fail loud rather than silently corrupt
        df = validate_merge(df, on=on)

    return df


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
    imputer = imputer_cls(df)

    dropped = imputer.drop_high_missingness(drop_threshold)
    if dropped:
        print(f"  Dropped {len(dropped)} column(s) exceeding {drop_threshold}% missingness: "
              f"{dropped}")

    if strategies:
        for strat in strategies:
            imputer.impute_columns(
                columns=strat.get('cols', []),
                method=strat.get('method', 'mean'),
                constant_val=strat.get('constant_val', None)
            )

    return imputer.get_dataframe()   # ← BUG FIX: was missing () in v2.0

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
    engineer = engineer_cls(df)

    tool_map = {
        'binarize':               engineer.binarize,
        'quantile_binning':       engineer.quantile_binning,
        'log_transform':          engineer.log_transform,
        'polynomial_interaction': engineer.polynomial_interaction,
        'custom_logic':           engineer.apply_custom_row_logic,
    }

    for step in steps:
        params    = step.copy()
        tool_name = params.pop('tool', None)

        if tool_name in tool_map:
            tool_map[tool_name](**params)
            print(f"  Engineered feature using '{tool_name}'.")
        else:
            print(f"  WARNING: Unknown tool '{tool_name}'. Skipping.")

    return engineer.get_dataframe()


#  STAGE 9 — Format for ML

def format_for_ml(
    df: pd.DataFrame,
    encode_categorical: bool = True,
    exclude_encode_prefixes: tuple = ('ID', 'LNK_', 'SITE'),
    scale_method: str = 'standard',        # 'standard', 'minmax', 'robust', or None
    exclude_scale_cols: list = None,       # Columns to protect from scaling (e.g. sequence keys)
    formatter_cls=ModelFormatter
) -> pd.DataFrame:
    """
    Final ML-readiness pass:
      • One-hot encode categorical columns (skip ID-like columns).
      • Scale numeric columns (skip explicit exclusions).
    """
    print(_header("Formatting for ML"))
    formatter = formatter_cls(df)

    if encode_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols
                    if not str(c).upper().startswith(tuple(p.upper() for p in exclude_encode_prefixes))]
        if cat_cols:
            formatter.encode_categorical(columns=cat_cols, drop_first=True)
            print(f"  One-hot encoded {len(cat_cols)} column(s).")

    if scale_method:
        protected = set(exclude_scale_cols or []) | {'ID', 'SEQNO'}
        num_cols = [c for c in df.select_dtypes(include=['int64', 'float64']).columns
                    if c not in protected]
        if num_cols:
            formatter.scale_numeric(columns=num_cols, method=scale_method)
            print(f"  Scaled {len(num_cols)} column(s) using '{scale_method}'.")

    return formatter.get_dataframe()


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


#  DEFAULT PIPELINE  ← the one function you call

def run_default_pipeline(
    primary_file: str,
    output_file: str = 'processed_output.csv',

    # --- Stage 2: text cleaning ---
    drop_columns: list = None,
    string_clean_cols: list = None,          # None = all string columns
    string_clean_options: dict = None,       # None = universal scrub + lowercase + strip

    # --- Stage 3: merges ---
    merge_configs: list = None,

    # --- Stage 4: merge collision resolution ---
    collision_strategy: str = 'keep_base',

    # --- Stage 5: deduplication ---
    dedup_subset: list = None,
    dedup_keep: str = 'first',
    dedup_sort_by: list = None,
    dedup_sort_ascending: bool = True,

    # --- Stage 6: lab unit normalisation ---
    lab_configs: list = None,

    # --- Stage 7: imputation ---
    impute_drop_threshold: float = 50.0,
    impute_strategies: list = None,

    # --- Stage 8: feature engineering ---
    engineer_steps: list = None,

    # --- Stage 9: ML formatting ---
    encode_categorical: bool = True,
    exclude_encode_prefixes: tuple = ('ID', 'LNK_', 'SITE'),
    scale_method: str = 'standard',
    exclude_scale_cols: list = None,

    # --- Dependency injection (for testing / subclassing) ---
    loader_cls=DataLoader,
    merger_cls=DataMerger,
    cleaner_cls=DataCleaner,
    imputer_cls=Imputer,
    engineer_cls=FeatureEngineer,
    formatter_cls=ModelFormatter,
) -> pd.DataFrame:
    """
    Run the full default processing pipeline end-to-end.

    Calling with only `primary_file` and `output_file` is enough for a
    standard run.  Every other parameter has a safe default.

    If you need a non-standard sequence (e.g. engineer features before
    imputing, or run two separate cleaning passes) call the individual
    stage functions directly — they all accept and return plain DataFrames.

    Minimal example
    ---------------
    df = run_default_pipeline('cohort.csv', 'cohort_ml_ready.csv')

    With merges and lab normalisation
    -----------------------------------
    df = run_default_pipeline(
        primary_file='cohort.csv',
        output_file='cohort_ml_ready.csv',
        merge_configs=[
            {'file': 'labs.csv', 'on': ['PATIENT_ID'],
             'filters': {'STATUS': 'Final'}, 'collapse_duplicates': True},
        ],
        lab_configs=[
            {'value_col': 'GLUCOSE', 'unit_col': 'GLUCOSE_UNIT',
             'canonical': 'mg/dL', 'conversions': {'mmol/L': 6.945}},
        ],
        impute_strategies=[
            {'cols': ['AGE', 'BMI'],  'method': 'median'},
            {'cols': ['SITE_CODE'],   'method': 'mode'},
        ],
        engineer_steps=[
            {'tool': 'log_transform', 'column': 'LOS_DAYS'},
        ],
    )
    """

    # 1. Load
    df = load_data(primary_file, loader_cls=loader_cls)

    # 2. Clean text on primary BEFORE any merge
    df = clean_text(
        df,
        columns=string_clean_cols,
        options=string_clean_options,
        drop_columns=drop_columns,
        cleaner_cls=cleaner_cls
    )

    # 3. Merge (each incoming dataset is also cleaned inside merge_datasets)
    if merge_configs:
        df = merge_datasets(
            df,
            merge_configs=merge_configs,
            loader_cls=loader_cls,
            merger_cls=merger_cls,
            cleaner_cls=cleaner_cls
        )

    # 4. Validate merge (resolve _x/_y collisions if any slipped through)
    df = validate_merge(df, collision_strategy=collision_strategy)

    # 5. Deduplicate
    df = deduplicate(
        df,
        subset=dedup_subset,
        keep=dedup_keep,
        sort_by=dedup_sort_by,
        sort_ascending=dedup_sort_ascending
    )

    # 6. Normalise lab units
    df = normalize_lab_units(df, lab_configs=lab_configs)

    # 7. Impute
    df = impute_missing(
        df,
        drop_threshold=impute_drop_threshold,
        strategies=impute_strategies,
        imputer_cls=imputer_cls
    )

    # 8. Feature engineering
    df = engineer_features(df, steps=engineer_steps, engineer_cls=engineer_cls)

    # 9. Format for ML
    df = format_for_ml(
        df,
        encode_categorical=encode_categorical,
        exclude_encode_prefixes=exclude_encode_prefixes,
        scale_method=scale_method,
        exclude_scale_cols=exclude_scale_cols,
        formatter_cls=formatter_cls
    )

    # 10. Save
    df = save_output(df, output_file=output_file)

    return df


def _header(title: str) -> str:
    return f"\n{'='*40}\n{title}\n{'='*40}"