import pandas as pd
import numpy as np


DEFAULT_ML_OPTIONS = {
    "one_hot": True,
    "unix_dates": True,
    "scale_numerics": False,
}


def prepare_for_ml(df, options=None):
    """Transform cleaned df into training-ready numeric-only df."""
    opts = {**DEFAULT_ML_OPTIONS, **(options or {})}
    df_ml = df.copy()
    log = []

    # Drop flag columns (audit only, not features)
    flag_cols = [c for c in df_ml.columns if c.endswith("_is_outlier")]
    if flag_cols:
        df_ml = df_ml.drop(columns=flag_cols)
        log.append(f"Dropped {len(flag_cols)} outlier flag columns")

    # Convert date columns to Unix timestamps
    if opts["unix_dates"]:
        for col in df_ml.columns:
            if df_ml[col].dtype == 'object':
                try:
                    conv = pd.to_datetime(df_ml[col], errors='coerce')
                    if conv.notna().sum() > len(df_ml) * 0.5:
                        df_ml[col] = conv.astype('int64') // 10**9
                        df_ml[col] = df_ml[col].replace(-2208988800, np.nan)  # NaT → nan
                        log.append(f"Converted '{col}' → Unix timestamp")
                except Exception:
                    pass

    # One-hot encode remaining objects
    cat_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        if opts["one_hot"]:
            df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True, dtype=int)
            log.append(f"One-hot encoded: {', '.join(cat_cols)}")
        else:
            df_ml = df_ml.drop(columns=cat_cols)
            log.append(f"Dropped categorical (one-hot off): {', '.join(cat_cols)}")

    # Handle remaining nulls (safety)
    for col in df_ml.select_dtypes(include=[np.number]).columns:
        if df_ml[col].isnull().sum() > 0:
            df_ml[col] = df_ml[col].fillna(df_ml[col].median())

    # Optional scaling
    if opts["scale_numerics"]:
        from sklearn.preprocessing import StandardScaler
        num_cols = df_ml.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df_ml[num_cols] = StandardScaler().fit_transform(df_ml[num_cols])
            log.append(f"Scaled {len(num_cols)} numeric columns")

    # Final assertion: must be all numeric
    non_numeric = df_ml.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        df_ml = df_ml.drop(columns=non_numeric)
        log.append(f"Dropped non-numeric leftovers: {', '.join(non_numeric)}")

    return df_ml, log