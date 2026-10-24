import streamlit as st
import pandas as pd
import time
import os
import tempfile

from core.cleaner import clean_dataset, DEFAULT_OPTIONS
from core.ml_prep import prepare_for_ml, DEFAULT_ML_OPTIONS
from core.plotting import make_boxplot, make_missing_bar
from core.report import generate_pdf_report

st.set_page_config(page_title="DataDoctor", layout="wide", page_icon="🩺")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Cleaning Options")

    st.subheader("📅 Dates")
    opt_standardize_dates = st.checkbox("Standardize to YYYY-MM-DD", value=True)
    opt_impute_dates = st.checkbox("Impute unparseable dates (mode)", value=True)

    st.subheader("🔢 Outliers")
    opt_flag_outliers = st.checkbox("Flag outliers (IQR)", value=True)
    opt_cap_outliers = st.checkbox("Cap outliers in-place", value=True)
    if not opt_cap_outliers:
        st.caption("⚠️ Original values kept + `_is_outlier` flag")

    st.subheader("🕳️ Missing Values")
    opt_impute_numeric = st.checkbox("Impute numeric (median)", value=True)
    opt_impute_categorical = st.checkbox("Impute categorical (mode)", value=True)
    opt_drop_high_null = st.checkbox("Drop columns >80% null", value=False)

    st.subheader("🏷️ Headers & Rows")
    opt_fix_headers = st.checkbox("Standardize column names", value=True)
    opt_strip_whitespace = st.checkbox("Strip whitespace", value=True)
    opt_remove_duplicates = st.checkbox("Remove duplicate rows", value=True)

    st.divider()
    st.subheader("🤖 Training Export")
    opt_one_hot = st.checkbox("One-hot encode categoricals", value=True)
    opt_unix_dates = st.checkbox("Unix timestamps for dates", value=True)
    opt_scale = st.checkbox("Standard-scale numerics", value=False)

    cleaning_options = {
        "fix_headers": opt_fix_headers,
        "strip_whitespace": opt_strip_whitespace,
        "remove_duplicates": opt_remove_duplicates,
        "standardize_dates": opt_standardize_dates,
        "impute_dates": opt_impute_dates,
        "impute_numeric": opt_impute_numeric,
        "impute_categorical": opt_impute_categorical,
        "drop_high_null": opt_drop_high_null,
        "flag_outliers": opt_flag_outliers,
        "cap_outliers": opt_cap_outliers,
    }
    ml_options = {
        "one_hot": opt_one_hot,
        "unix_dates": opt_unix_dates,
        "scale_numerics": opt_scale,
    }

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>🩺 DataDoctor</h1>
<p style='text-align:center; color:gray;'>Automated CSV Profiler & Cleaning Pipeline</p>
<hr>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload messy CSV or Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    start = time.time()
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Cannot read file: {e}")
        st.stop()

    # --- Empty-file guard ---
    if df is None or df.empty or len(df.columns) == 0:
        st.error("❌ The uploaded file is empty (0 rows / 0 columns). Upload a dataset with data.")
        st.stop()
    if df.dropna(how='all').empty:
        st.warning("⚠️ Every row is empty. Nothing to clean.")
        st.stop()

    raw_df = df.copy()
    clean_df, stats, history, column_report = clean_dataset(df, cleaning_options)
    ml_df, ml_log = prepare_for_ml(clean_df, ml_options)
    elapsed = time.time() - start

    # Filename base
    base = os.path.splitext(uploaded_file.name)[0]

    # ---------- TOP METRICS (colorful) ----------
    score = stats['health_score']
    if score >= 80:
        score_color, score_bg, score_label = "#0f9d58", "#e6f4ea", "Excellent"
    elif score >= 50:
        score_color, score_bg, score_label = "#f4b400", "#fef7e0", "Needs Work"
    else:
        score_color, score_bg, score_label = "#db4437", "#fce8e6", "Poor"

    st.markdown(f"""
    <style>
    .metric-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 16px 0; }}
    .metric-card {{
        padding: 18px; border-radius: 12px; text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08); border: 1px solid #eee;
    }}
    .metric-value {{ font-size: 26px; font-weight: 700; margin: 0; }}
    .metric-label {{ font-size: 12px; color: #666; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.5px; }}
    </style>
    <div class="metric-grid">
        <div class="metric-card" style="background:{score_bg}; border-color:{score_color};">
            <p class="metric-value" style="color:{score_color};">{score:.0f}/100</p>
            <p class="metric-label" style="color:{score_color};">Health • {score_label}</p>
        </div>
        <div class="metric-card" style="background:#e8f0fe; border-color:#4285f4;">
            <p class="metric-value" style="color:#1a73e8;">{(stats['missing_fixed']/max(raw_df.size,1)*100):.1f}%</p>
            <p class="metric-label" style="color:#1a73e8;">Missing</p>
        </div>
        <div class="metric-card" style="background:#fce8e6; border-color:#db4437;">
            <p class="metric-value" style="color:#c5221f;">{stats['duplicates_removed']}</p>
            <p class="metric-label" style="color:#c5221f;">Duplicates</p>
        </div>
        <div class="metric-card" style="background:#fef7e0; border-color:#f4b400;">
            <p class="metric-value" style="color:#b8860b;">{stats['outliers_flagged']}</p>
            <p class="metric-label" style="color:#b8860b;">Outliers</p>
        </div>
        <div class="metric-card" style="background:#e6f4ea; border-color:#0f9d58;">
            <p class="metric-value" style="color:#0f9d58;">{ml_df.shape[1]}</p>
            <p class="metric-label" style="color:#0f9d58;">ML Features</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- BEFORE / AFTER ----------
    st.subheader("🔍 Before vs After")
    st.caption(f"Input: {raw_df.shape[0]} rows × {raw_df.shape[1]} cols   |   Cleaned: {clean_df.shape[0]} rows × {clean_df.shape[1]} cols")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**❌ Input** — `{raw_df.shape[0]} rows`")
        st.dataframe(raw_df, use_container_width=True, height=500)
    with col2:
        st.markdown(f"**✅ Cleaned** — `{clean_df.shape[0]} rows`")
        st.dataframe(clean_df, use_container_width=True, height=500)

    # ---------- BOX-PLOTS ----------
    numeric_cols = clean_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.endswith("_is_outlier")]

    boxplot_paths = []
    if numeric_cols:
        st.subheader("📦 Outlier Box-Plots (IQR)")
        cols = st.columns(min(3, len(numeric_cols)))
        for i, col in enumerate(numeric_cols[:6]):
            fig = make_boxplot(clean_df, col)
            with cols[i % len(cols)]:
                st.pyplot(fig, use_container_width=True)
            # Save for PDF
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, dpi=80, bbox_inches="tight")
            boxplot_paths.append(tmp.name)

    # ---------- MISSING CHART ----------
    if column_report:
        st.subheader("📊 Missing Values by Column")
        st.pyplot(make_missing_bar(column_report), use_container_width=True)

    # ---------- CLEANING HISTORY ----------
    st.subheader("🧾 Cleaning History")
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, height=400)
    else:
        st.info("No cleaning actions taken.")

    # ---------- TRAINING PREP ----------
    st.subheader("🤖 Training-Ready Preview")
    st.caption(f"{ml_df.shape[0]} rows × {ml_df.shape[1]} numeric features — all numeric, no nulls")
    st.dataframe(ml_df, use_container_width=True, height=400)
    if ml_log:
        with st.expander("Transformation log"):
            for line in ml_log:
                st.write(f"• {line}")

    # ---------- DOWNLOADS ----------
    st.divider()
    st.subheader("📦 Downloads")

    col_a, col_b = st.columns(2)
    col_a.download_button("✅ Training CSV", ml_df.to_csv(index=False),
                          f"datadoctor_{base}.csv", type="primary", use_container_width=True)

    # Excel training
    import io
    xlsx_buf = io.BytesIO()
    ml_df.to_excel(xlsx_buf, index=False, engine='openpyxl')
    col_b.download_button("✅ Training XLSX", xlsx_buf.getvalue(),
                          f"datadoctor_{base}.xlsx", use_container_width=True)

    col_c, col_d = st.columns(2)
    col_c.download_button("📄 Human Cleaned CSV", clean_df.to_csv(index=False),
                          f"datadoctor_cleaned_{base}.csv", use_container_width=True)

    pdf_bytes = generate_pdf_report(stats, history, uploaded_file.name, boxplot_paths)
    col_d.download_button("📄 PDF Report", pdf_bytes, f"datadoctor_{base}_report.pdf", use_container_width=True)