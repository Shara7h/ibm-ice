import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _apply_dark_style(fig, ax):
    """Apply modern dark-theme styling to matplotlib figure and axis."""
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#1E293B')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    ax.xaxis.label.set_color('#94A3B8')
    ax.yaxis.label.set_color('#94A3B8')
    ax.title.set_color('#F8FAFC')
    ax.grid(color='#334155', linestyle='--', linewidth=0.7, alpha=0.5)


def make_boxplot(df, col):
    """Return a modern dark-themed box-plot for one numeric column."""
    fig, ax = plt.subplots(figsize=(5, 3))
    _apply_dark_style(fig, ax)

    data = df[col].dropna()
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', color='#94A3B8')
        return fig

    ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="#6366F1", color="#818CF8", alpha=0.75, linewidth=1.5),
        whiskerprops=dict(color="#94A3B8", linewidth=1.2),
        capprops=dict(color="#94A3B8", linewidth=1.2),
        medianprops=dict(color="#38BDF8", linewidth=2.5),
        flierprops=dict(marker='o', markerfacecolor="#F43F5E", markeredgecolor="none", markersize=5, alpha=0.8)
    )
    ax.set_title(f"{col}", fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel("Value", fontsize=9)
    fig.tight_layout()
    return fig


def make_missing_bar(column_report):
    """Return a modern horizontal bar chart of missing values by column."""
    fig, ax = plt.subplots(figsize=(6, 3))
    _apply_dark_style(fig, ax)

    if not column_report:
        ax.text(0.5, 0.5, "No missing values detected", ha='center', va='center', color='#10B981', fontweight='bold')
        return fig

    cols = list(column_report.keys())
    missing = [column_report[c]["missing"] for c in cols]

    bars = ax.barh(cols, missing, color="#F43F5E", alpha=0.85, edgecolor="#FB7185", linewidth=1, height=0.55)
    
    # Add count labels at end of bars
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            ax.text(width + max(missing) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{int(width)}', va='center', ha='left', color='#F8FAFC', fontsize=8, fontweight='bold')

    ax.set_xlabel("Missing Count", fontsize=9)
    ax.set_title("Missing Values by Column", fontsize=11, fontweight='bold', pad=10)
    fig.tight_layout()
    return fig