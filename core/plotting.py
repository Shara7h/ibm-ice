import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_boxplot(df, col):
    """Return a matplotlib figure of a box-plot for one numeric column."""
    fig, ax = plt.subplots(figsize=(5, 3))
    data = df[col].dropna()
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha='center')
        return fig
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#4A90E2", alpha=0.6),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(f"{col}", fontsize=10)
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_missing_bar(column_report):
    """Bar chart of missing % per column."""
    fig, ax = plt.subplots(figsize=(6, 3))
    if not column_report:
        ax.text(0.5, 0.5, "No issues", ha='center')
        return fig
    cols = list(column_report.keys())
    missing = [column_report[c]["missing"] for c in cols]
    ax.barh(cols, missing, color="#E24A4A")
    ax.set_xlabel("Missing values")
    ax.set_title("Missing by Column")
    fig.tight_layout()
    return fig