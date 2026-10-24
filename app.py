import streamlit as st
import pandas as pd
import time
import os
import tempfile
import io

from core.cleaner import clean_dataset, DEFAULT_OPTIONS
from core.ml_prep import prepare_for_ml, DEFAULT_ML_OPTIONS
from core.plotting import make_boxplot, make_missing_bar
from core.report import generate_pdf_report

st.set_page_config(
    page_title="DataDoctor · AI Data Sanitizer",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# ---------- GLOBAL CUSTOM CSS & DESIGN SYSTEM ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background gradient styling */
.stApp {
    background: radial-gradient(circle at 50% 0%, #172033 0%, #0B0F19 75%);
}

/* Hero Banner */
.hero-container {
    text-align: center;
    padding: 30px 20px 24px 20px;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0) 100%);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    margin-bottom: 28px;
    backdrop-filter: blur(16px);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    color: #A5B4FC;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #FFFFFF 20%, #CBD5E1 60%, #38BDF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    font-size: 15px;
    color: #94A3B8;
    margin-top: 8px;
    margin-bottom: 16px;
    font-weight: 400;
}

.hero-tags {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
}

.hero-tag {
    font-size: 12px;
    color: #CBD5E1;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 4px 12px;
    border-radius: 8px;
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin: 20px 0 28px 0;
}

.metric-card {
    padding: 20px 16px;
    border-radius: 16px;
    text-align: center;
    background: rgba(17, 24, 39, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.5);
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--card-accent, #6366F1);
}

.metric-icon-bubble {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin-bottom: 10px;
    background: var(--card-bg, rgba(99, 102, 241, 0.15));
    border: 1px solid var(--card-border, rgba(99, 102, 241, 0.3));
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
    color: #F8FAFC;
    line-height: 1.1;
}

.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #94A3B8;
    margin: 6px 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.75px;
}

.metric-sub {
    font-size: 11px;
    margin-top: 4px;
    font-weight: 500;
}

/* Section Header Badges */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #38BDF8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Card Containers */
.glass-panel {
    background: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(12px);
    margin-bottom: 24px;
}

/* Download Cards */
.download-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.download-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    background: rgba(30, 41, 59, 0.8);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #0F172A !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

/* Streamlit Native Overrides */
/* Streamlit File Uploader Overrides */
div[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 2px dashed #94A3B8 !important;
    border-radius: 16px !important;
    padding: 18px !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #4F46E5 !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC !important;
    border: 1px dashed #CBD5E1 !important;
    border-radius: 12px !important;
}

/* Explicitly style the Browse Files button */
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button[kind="secondary"],
div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
    background-color: #4F46E5 !important;
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: 1px solid #4338CA !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 8px 18px !important;
    box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
}

div[data-testid="stFileUploader"] button:hover,
div[data-testid="stFileUploader"] button[kind="secondary"]:hover,
div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover {
    background-color: #4338CA !important;
    background: #4338CA !important;
    color: #FFFFFF !important;
}

/* Make sure button label text is ALWAYS white */
div[data-testid="stFileUploader"] button *,
div[data-testid="stFileUploader"] button p,
div[data-testid="stFileUploader"] button span,
div[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] * {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Dropzone instructions text (Drag and drop file here, limit 200MB) */
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #334155 !important;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}

.stDownloadButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
        <span style="font-size:26px;">🩺</span>
        <div>
            <div style="font-size:18px; font-weight:800; color:#F8FAFC; line-height:1.2;">DataDoctor</div>
            <div style="font-size:11px; color:#94A3B8; font-weight:500;">PIPELINE CONTROLS</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

# ---------- HERO HEADER ----------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">✨ AI-POWERED SANITIZATION ENGINE v2.0</div>
    <h1 class="hero-title">🩺 DataDoctor</h1>
    <p class="hero-sub">Automated CSV Profiler, Robust Cleaning Pipeline & ML-Ready Feature Extractor</p>
    <div class="hero-tags">
        <span class="hero-tag">⚡ Sub-Second Profiling</span>
        <span class="hero-tag">📅 ISO 8601 Standardization</span>
        <span class="hero-tag">🛡️ Zero Data Leakage</span>
        <span class="hero-tag">🤖 Instant ML Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop your messy CSV or Excel file here to automatically cleanse and profile", type=['csv', 'xlsx', 'xls'])

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

    # ---------- TOP METRICS (Premium Glassmorphic Cards) ----------
    score = stats['health_score']
    if score >= 80:
        score_accent = "#10B981"
        score_bg = "rgba(16, 185, 129, 0.12)"
        score_border = "rgba(16, 185, 129, 0.3)"
        score_label = "OPTIMAL"
        score_icon = "🟢"
    elif score >= 50:
        score_accent = "#F59E0B"
        score_bg = "rgba(245, 158, 11, 0.12)"
        score_border = "rgba(245, 158, 11, 0.3)"
        score_label = "NEEDS WORK"
        score_icon = "🟡"
    else:
        score_accent = "#F43F5E"
        score_bg = "rgba(244, 63, 94, 0.12)"
        score_border = "rgba(244, 63, 94, 0.3)"
        score_label = "CRITICAL"
        score_icon = "🔴"

    missing_pct = (stats['missing_fixed'] / max(raw_df.size, 1) * 100)

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card" style="--card-accent:{score_accent}; --card-bg:{score_bg}; --card-border:{score_border}; border-color:{score_border};">
            <div class="metric-icon-bubble">{score_icon}</div>
            <p class="metric-value" style="color:{score_accent};">{score:.0f}<span style="font-size:16px; font-weight:600; opacity:0.8;">/100</span></p>
            <p class="metric-label">Health Score</p>
            <p class="metric-sub" style="color:{score_accent};">{score_label}</p>
        </div>
        <div class="metric-card" style="--card-accent:#38BDF8; --card-bg:rgba(56, 189, 248, 0.12); --card-border:rgba(56, 189, 248, 0.3);">
            <div class="metric-icon-bubble">🕳️</div>
            <p class="metric-value" style="color:#38BDF8;">{missing_pct:.1f}%</p>
            <p class="metric-label">Missing Values</p>
            <p class="metric-sub" style="color:#94A3B8;">{stats['missing_fixed']} Imputed</p>
        </div>
        <div class="metric-card" style="--card-accent:#EC4899; --card-bg:rgba(236, 72, 153, 0.12); --card-border:rgba(236, 72, 153, 0.3);">
            <div class="metric-icon-bubble">✂️</div>
            <p class="metric-value" style="color:#EC4899;">{stats['duplicates_removed']}</p>
            <p class="metric-label">Duplicates</p>
            <p class="metric-sub" style="color:#94A3B8;">Rows Purged</p>
        </div>
        <div class="metric-card" style="--card-accent:#F59E0B; --card-bg:rgba(245, 158, 11, 0.12); --card-border:rgba(245, 158, 11, 0.3);">
            <div class="metric-icon-bubble">🎯</div>
            <p class="metric-value" style="color:#F59E0B;">{stats['outliers_flagged']}</p>
            <p class="metric-label">Outliers</p>
            <p class="metric-sub" style="color:#94A3B8;">IQR Capped</p>
        </div>
        <div class="metric-card" style="--card-accent:#8B5CF6; --card-bg:rgba(139, 92, 246, 0.12); --card-border:rgba(139, 92, 246, 0.3);">
            <div class="metric-icon-bubble">🤖</div>
            <p class="metric-value" style="color:#A78BFA;">{ml_df.shape[1]}</p>
            <p class="metric-label">ML Features</p>
            <p class="metric-sub" style="color:#94A3B8;">Ready for Fit</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- BEFORE / AFTER ----------
    st.markdown("""
    <div>
        <div class="section-badge">01 / INSPECTION</div>
        <h2 class="section-title">🔍 Before vs After Purification</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Raw Input: **{raw_df.shape[0]} rows × {raw_df.shape[1]} cols**  →  Purified: **{clean_df.shape[0]} rows × {clean_df.shape[1]} cols** (Processed in {elapsed:.2f}s)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.25); border-radius:10px; padding:8px 14px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#FCA5A5; font-weight:700; font-size:13px;">❌ RAW DIRTY INPUT</span>
            <span style="color:#94A3B8; font-size:12px; font-family:'JetBrains Mono';">{raw_df.shape[0]} rows</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(raw_df, use_container_width=True, height=450)

    with col2:
        st.markdown(f"""
        <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.25); border-radius:10px; padding:8px 14px; margin-bottom:10px; display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#6EE7B7; font-weight:700; font-size:13px;">✅ PURIFIED DATASET (YYYY-MM-DD)</span>
            <span style="color:#94A3B8; font-size:12px; font-family:'JetBrains Mono';">{clean_df.shape[0]} rows</span>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(clean_df, use_container_width=True, height=450)

    st.write("")

    # ---------- BOX-PLOTS & MISSING VISUALIZATION ----------
    numeric_cols = clean_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.endswith("_is_outlier")]

    boxplot_paths = []
    if numeric_cols or column_report:
        st.markdown("""
        <div>
            <div class="section-badge">02 / DISTRIBUTION</div>
            <h2 class="section-title">📊 Anomaly & Integrity Charts</h2>
        </div>
        """, unsafe_allow_html=True)

        chart_col_left, chart_col_right = st.columns([1, 1])

        with chart_col_left:
            if column_report:
                st.markdown("<p style='font-size:13px; font-weight:600; color:#CBD5E1; margin-bottom:6px;'>Missing Values Frequency</p>", unsafe_allow_html=True)
                st.pyplot(make_missing_bar(column_report), use_container_width=True)

        with chart_col_right:
            if numeric_cols:
                st.markdown("<p style='font-size:13px; font-weight:600; color:#CBD5E1; margin-bottom:6px;'>Outlier Boxplot (IQR Analysis)</p>", unsafe_allow_html=True)
                fig = make_boxplot(clean_df, numeric_cols[0])
                st.pyplot(fig, use_container_width=True)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(tmp.name, dpi=80, bbox_inches="tight")
                boxplot_paths.append(tmp.name)

        if len(numeric_cols) > 1:
            with st.expander("🔍 View All Numeric Outlier Box-Plots"):
                more_cols = st.columns(min(3, len(numeric_cols) - 1))
                for i, col in enumerate(numeric_cols[1:7]):
                    fig_i = make_boxplot(clean_df, col)
                    with more_cols[i % len(more_cols)]:
                        st.pyplot(fig_i, use_container_width=True)
                    tmp_i = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    fig_i.savefig(tmp_i.name, dpi=80, bbox_inches="tight")
                    boxplot_paths.append(tmp_i.name)

    st.write("")

    # ---------- CLEANING HISTORY AUDIT LOG ----------
    st.markdown("""
    <div>
        <div class="section-badge">03 / AUDIT LOG</div>
        <h2 class="section-title">🧾 Comprehensive Audit Trail</h2>
    </div>
    """, unsafe_allow_html=True)

    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, height=350)
    else:
        st.info("No cleaning transformations were necessary — dataset is pristine.")

    st.write("")

    # ---------- MACHINE LEARNING PREP ----------
    st.markdown("""
    <div>
        <div class="section-badge">04 / MACHINE LEARNING</div>
        <h2 class="section-title">🤖 Model-Ready Numeric Feature Matrix</h2>
    </div>
    """, unsafe_allow_html=True)
    st.caption(f"Fully Encoded & Scaled: **{ml_df.shape[0]} rows × {ml_df.shape[1]} numeric columns** — 0 nulls, model.fit() ready")
    st.dataframe(ml_df, use_container_width=True, height=350)

    if ml_log:
        with st.expander("🛠️ View Feature Transformation Log"):
            for line in ml_log:
                st.write(f"• `{line}`")

    # ---------- DOWNLOADS ----------
    st.write("")
    st.markdown("""
    <div>
        <div class="section-badge">05 / EXPORT</div>
        <h2 class="section-title">📦 Production Artifact Downloads</h2>
    </div>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.download_button(
            "⚡ Training CSV",
            ml_df.to_csv(index=False),
            f"datadoctor_ml_{base}.csv",
            type="primary",
            use_container_width=True,
            help="All numeric, one-hot encoded, scaled features"
        )

    with d2:
        xlsx_buf = io.BytesIO()
        ml_df.to_excel(xlsx_buf, index=False, engine='openpyxl')
        st.download_button(
            "📊 Training XLSX",
            xlsx_buf.getvalue(),
            f"datadoctor_ml_{base}.xlsx",
            use_container_width=True,
            help="Excel workbook with ML ready features"
        )

    with d3:
        st.download_button(
            "📄 Cleaned Data CSV",
            clean_df.to_csv(index=False),
            f"datadoctor_clean_{base}.csv",
            use_container_width=True,
            help="Human-readable dataset with standardized dates (YYYY-MM-DD)"
        )

    with d4:
        pdf_bytes = generate_pdf_report(stats, history, uploaded_file.name, boxplot_paths)
        st.download_button(
            "📑 Audit PDF Report",
            pdf_bytes,
            f"datadoctor_{base}_audit.pdf",
            use_container_width=True,
            help="Complete PDF report with metrics and visualizations"
        )