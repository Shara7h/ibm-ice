import pandas as pd
import numpy as np


DEFAULT_OPTIONS = {
    "fix_headers": True,
    "strip_whitespace": True,
    "remove_duplicates": True,
    "standardize_dates": True,
    "impute_dates": True,
    "impute_numeric": True,
    "impute_categorical": True,
    "drop_high_null": False,
    "flag_outliers": True,
    "cap_outliers": True,
}

DATE_HINTS = ('date', 'time', 'created', 'updated', 'purchase', 'dob',
              'hire', 'birth', '_at', '_on', 'timestamp', 'day', 'month', 'year')


def _looks_like_date_col(series, col_name):
    """Return True if column should be treated as a date column."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    converted = pd.to_datetime(non_null, format='mixed', errors='coerce')
    valid_ratio = converted.notna().sum() / len(non_null)
    has_hint = any(h in col_name.lower() for h in DATE_HINTS)

    # Strong name hint → always treat as date (even if some garbage)
    if has_hint:
        return True
    # No hint → require high parse ratio
    return valid_ratio >= 0.9


def clean_dataset(df, options=None):
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    stats = {}
    history = []
    column_report = {}
    raw_rows = len(df)
    raw_cols = list(df.columns)

    # --- 1. Headers ---
    if opts["fix_headers"] or opts["strip_whitespace"]:
        changes = []
        new_cols = []
        for c in df.columns:
            fixed = str(c).strip()
            if opts["fix_headers"]:
                fixed = fixed.replace(' ', '_').replace('.', '')
            if opts["strip_whitespace"]:
                fixed = fixed.strip()
            if fixed != str(c):
                changes.append(f"'{c}'→'{fixed}'")
            new_cols.append(fixed)
        df.columns = new_cols
        if changes:
            history.append({
                "action": "Fix Headers", "column": "ALL",
                "values_changed": len(changes),
                "reason": "Standardized column names",
                "before": "; ".join(changes[:3]),
                "after": "Cleaned"
            })
    else:
        history.append({"action": "Fix Headers", "column": "ALL",
                        "values_changed": 0, "reason": "Skipped (user disabled)",
                        "before": "-", "after": "-"})

    # --- 2. Dates ---
    dates_fixed = 0
    date_cols = []
    for col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        if not _looks_like_date_col(df[col], col):
            continue
        date_cols.append(col)
        if opts["standardize_dates"]:
            before_vals = df[col].dropna().astype(str).head(2).tolist()
            converted = pd.to_datetime(df[col], format='mixed', errors='coerce')
            df[col] = converted.dt.strftime("%Y-%m-%d")
            dates_fixed += 1
            history.append({"action": "Standardize Date", "column": col,
                            "values_changed": int(converted.notna().sum()),
                            "reason": "Detected date column → YYYY-MM-DD",
                            "before": ", ".join(before_vals),
                            "after": ", ".join(df[col].dropna().head(2).tolist())})
        if opts["impute_dates"] and df[col].isnull().sum() > 0:
            nulls = int(df[col].isnull().sum())
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "1970-01-01"
            df[col] = df[col].fillna(fill_val)
            history.append({"action": "Impute Dates", "column": col,
                            "values_changed": nulls,
                            "reason": "Unparseable → mode date",
                            "before": f"{nulls} nulls", "after": f"Filled '{fill_val}'"})

    # --- 3. Missing values (numeric + categorical) ---
    missing_fixed = 0
    for col in df.columns:
        nulls = int(df[col].isnull().sum())
        if nulls == 0 or col in date_cols:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if opts["impute_numeric"]:
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                missing_fixed += nulls
                history.append({"action": "Impute Missing", "column": col,
                                "values_changed": nulls, "reason": "Numeric nulls",
                                "before": f"{nulls} nulls", "after": f"Median ({fill:.2f})"})
        else:
            if opts["impute_categorical"]:
                mode_val = df[col].mode()
                fill = mode_val[0] if not mode_val.empty else "Unknown"
                df[col] = df[col].fillna(fill)
                missing_fixed += nulls
                history.append({"action": "Impute Missing", "column": col,
                                "values_changed": nulls, "reason": "Categorical nulls",
                                "before": f"{nulls} nulls", "after": f"Mode ('{fill}')"})

    # --- 4. Drop high-null columns (optional) ---
    if opts["drop_high_null"]:
        for col in list(df.columns):
            if df[col].isnull().sum() / len(df) > 0.8:
                pct = df[col].isnull().sum() / len(df) * 100
                df = df.drop(columns=[col])
                history.append({"action": "Drop Column", "column": col,
                                "values_changed": 1, "reason": f"{pct:.0f}% null",
                                "before": "present", "after": "dropped"})

    # --- 5. Drop Duplicates (runs after cleaning) ---
    dupes = 0
    if opts["remove_duplicates"]:
        # Normalize once more: strip text for comparison
        df_for_compare = df.copy()
        for col in df_for_compare.select_dtypes(include=['object', 'string']).columns:
            df_for_compare[col] = df_for_compare[col].astype(str).str.strip()

        mask = df_for_compare.duplicated()
        dupes = int(mask.sum())
        if dupes:
            df = df[~mask.values].reset_index(drop=True)
            history.append({"action": "Remove Duplicates", "column": "ALL",
                            "values_changed": dupes, "reason": "Exact duplicates (post-cleaning)",
                            "before": f"{dupes} rows", "after": "Removed"})
    else:
        history.append({"action": "Remove Duplicates", "column": "ALL",
                        "values_changed": 0, "reason": "Skipped (user disabled)",
                        "before": "-", "after": "-"})
    stats["duplicates_removed"] = dupes

    # --- 6. Outliers ---
    outliers_total = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (df[col] < lower) | (df[col] > upper)
        n = int(mask.sum())
        if n == 0:
            continue
        outliers_total += n
        sample = df.loc[mask, col].head(2).tolist()

        if opts["flag_outliers"]:
            df[f"{col}_is_outlier"] = mask.astype(int)
        if opts["cap_outliers"]:
            df[col] = np.clip(df[col], lower, upper)
            action_note = f"Capped to [{lower:.1f}, {upper:.1f}]"
        else:
            action_note = f"Kept + flag column"

        history.append({"action": "Outliers", "column": col,
                        "values_changed": n,
                        "reason": f"IQR bounds {lower:.1f}–{upper:.1f}",
                        "before": ", ".join(map(str, sample)),
                        "after": action_note})

        column_report[col] = {
            "outliers": n, "lower": float(lower), "upper": float(upper),
            "type": "numeric",
            "missing": int(df[col].isnull().sum())
        }

    # --- Final stats ---
    stats.update({
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "raw_cols": len(raw_cols),
        "clean_cols": len(df.columns),
        "missing_fixed": missing_fixed,
        "outliers_flagged": outliers_total,
        "dates_fixed": dates_fixed,
        "date_cols": date_cols,
        "health_score": max(0, 100 - ((missing_fixed + dupes) / max(df.size, 1) * 100)),
    })
    return df, stats, history, column_report