"""
Basic analysis of MS2 spectra from LipidBlast parquet file.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Configure matplotlib for better figures
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# File paths
PARQUET_FILE = "MoNA-export-LipidBlast_2022.parquet"
OUTPUT_DIR = "figures"


def load_data(filepath: str) -> pd.DataFrame:
    """Load parquet file and add derived columns."""
    print(f"Loading data from {filepath}")
    df = pd.read_parquet(filepath)

    # Add derived columns
    df["n_peaks"] = df["mz_list"].apply(len)
    df["max_intensity"] = df["intensity_list"].apply(lambda x: max(x) if len(x) > 0 else 0)
    df["total_intensity"] = df["intensity_list"].apply(lambda x: sum(x) if len(x) > 0 else 0)

    # Extract lipid class from name (first word before space)
    df["lipid_class"] = df["name"].str.split().str[0]

    print(f"Loaded {len(df)} spectra")
    return df


def plot_precursor_mz_distribution(df: pd.DataFrame, output_path: str):
    """Plot precursor m/z distribution by ion mode."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram by ion mode
    for ax, mode in zip(axes, ["positive", "negative"]):
        subset = df[df["mode"] == mode]
        ax.hist(subset["precursor_mz"], bins=100, alpha=0.8,
                color="#1f77b4" if mode == "positive" else "#d62728",
                edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Precursor m/z")
        ax.set_ylabel("Count")
        ax.set_title(f"{mode.capitalize()} ion mode (n={len(subset):,})")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.suptitle("Precursor m/z Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_peaks_per_spectrum(df: pd.DataFrame, output_path: str):
    """Plot distribution of number of peaks per spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Overall distribution
    ax = axes[0]
    ax.hist(df["n_peaks"], bins=50, alpha=0.8, color="#2ca02c", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of peaks")
    ax.set_ylabel("Count")
    ax.set_title("All spectra")
    ax.axvline(df["n_peaks"].median(), color="red", linestyle="--", linewidth=1.5,
               label=f"Median: {df['n_peaks'].median():.0f}")
    ax.legend()

    # By ion mode (boxplot)
    ax = axes[1]
    mode_data = [df[df["mode"] == "positive"]["n_peaks"],
                 df[df["mode"] == "negative"]["n_peaks"]]
    bp = ax.boxplot(mode_data, labels=["Positive", "Negative"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#1f77b4")
    bp["boxes"][1].set_facecolor("#d62728")
    for box in bp["boxes"]:
        box.set_alpha(0.7)
    ax.set_ylabel("Number of peaks")
    ax.set_title("By ion mode")

    plt.suptitle("Peaks per Spectrum", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_lipid_class_distribution(df: pd.DataFrame, output_path: str, top_n: int = 20):
    """Plot distribution of lipid classes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top lipid classes
    class_counts = df["lipid_class"].value_counts().head(top_n)

    # Create horizontal bar plot
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_counts)))
    bars = ax.barh(range(len(class_counts)), class_counts.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(class_counts)))
    ax.set_yticklabels(class_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of spectra")
    ax.set_title(f"Top {top_n} Lipid Classes", fontsize=14, fontweight="bold")

    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
        ax.text(count + max(class_counts) * 0.01, i, f"{count:,}",
                va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_adduct_distribution(df: pd.DataFrame, output_path: str, top_n: int = 15):
    """Plot distribution of adduct types by ion mode."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mode in zip(axes, ["positive", "negative"]):
        subset = df[df["mode"] == mode]
        adduct_counts = subset["adduct_name"].value_counts().head(top_n)

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(adduct_counts))) if mode == "positive" \
                 else plt.cm.Reds(np.linspace(0.4, 0.9, len(adduct_counts)))

        bars = ax.barh(range(len(adduct_counts)), adduct_counts.values, color=colors)
        ax.set_yticks(range(len(adduct_counts)))
        ax.set_yticklabels(adduct_counts.index)
        ax.invert_yaxis()
        ax.set_xlabel("Number of spectra")
        ax.set_title(f"{mode.capitalize()} mode adducts")

        # Add count labels
        for i, count in enumerate(adduct_counts.values):
            ax.text(count + max(adduct_counts) * 0.01, i, f"{count:,}",
                    va="center", fontsize=8)

    plt.suptitle("Adduct Type Distribution", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_mz_vs_peaks(df: pd.DataFrame, output_path: str, sample_size: int = 10000):
    """Scatter plot of precursor m/z vs number of peaks."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Sample data if too large
    if len(df) > sample_size:
        plot_df = df.sample(sample_size, random_state=42)
    else:
        plot_df = df

    # Create scatter plot with color by ion mode
    for mode, color, marker in [("positive", "#1f77b4", "o"), ("negative", "#d62728", "s")]:
        subset = plot_df[plot_df["mode"] == mode]
        ax.scatter(subset["precursor_mz"], subset["n_peaks"],
                   c=color, alpha=0.3, s=10, label=f"{mode.capitalize()} (n={len(subset):,})",
                   marker=marker)

    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Number of peaks")
    ax.set_title("Precursor m/z vs. Fragment Count", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_intensity_distribution(df: pd.DataFrame, output_path: str):
    """Plot intensity distribution analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Sample spectra for intensity analysis
    sample_intensities = []
    for intensities in df["intensity_list"].sample(min(10000, len(df)), random_state=42):
        sample_intensities.extend(intensities)
    sample_intensities = np.array(sample_intensities)

    # 1. Raw intensity distribution (log scale)
    ax = axes[0, 0]
    ax.hist(sample_intensities[sample_intensities > 0], bins=100, alpha=0.8,
            color="#9467bd", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Count")
    ax.set_title("Peak intensity distribution")
    ax.set_yscale("log")

    # 2. Normalized intensity distribution
    ax = axes[0, 1]
    # Normalize each spectrum to max=100
    normalized_intensities = []
    for intensities in df["intensity_list"].sample(min(10000, len(df)), random_state=42):
        if max(intensities) > 0:
            normalized = [i / max(intensities) * 100 for i in intensities]
            normalized_intensities.extend(normalized)
    ax.hist(normalized_intensities, bins=50, alpha=0.8, color="#17becf",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Relative intensity (%)")
    ax.set_ylabel("Count")
    ax.set_title("Normalized intensity distribution")

    # 3. Total intensity per spectrum by lipid class (top 10 classes)
    ax = axes[1, 0]
    top_classes = df["lipid_class"].value_counts().head(10).index
    class_data = [df[df["lipid_class"] == c]["total_intensity"] for c in top_classes]
    bp = ax.boxplot(class_data, labels=top_classes, patch_artist=True, vert=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_classes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(top_classes, rotation=45, ha="right")
    ax.set_ylabel("Total intensity")
    ax.set_title("Total intensity by lipid class")

    # 4. Max intensity per spectrum histogram
    ax = axes[1, 1]
    ax.hist(df["max_intensity"], bins=50, alpha=0.8, color="#8c564b",
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Max intensity per spectrum")
    ax.set_ylabel("Count")
    ax.set_title("Base peak intensity distribution")

    plt.suptitle("Intensity Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_mz_heatmap(df: pd.DataFrame, output_path: str, sample_size: int = 5000):
    """2D histogram of precursor m/z vs fragment m/z."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mode in zip(axes, ["positive", "negative"]):
        subset = df[df["mode"] == mode]
        if len(subset) > sample_size:
            subset = subset.sample(sample_size, random_state=42)

        # Collect precursor-fragment pairs
        precursor_mzs = []
        fragment_mzs = []
        for _, row in subset.iterrows():
            for frag_mz in row["mz_list"]:
                precursor_mzs.append(row["precursor_mz"])
                fragment_mzs.append(frag_mz)

        # Create 2D histogram
        h = ax.hist2d(precursor_mzs, fragment_mzs, bins=100,
                      cmap="viridis" if mode == "positive" else "magma",
                      norm=plt.matplotlib.colors.LogNorm())
        plt.colorbar(h[3], ax=ax, label="Count")

        # Add diagonal line (precursor = fragment)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'w--', alpha=0.5, linewidth=1, label="Precursor = Fragment")

        ax.set_xlabel("Precursor m/z")
        ax.set_ylabel("Fragment m/z")
        ax.set_title(f"{mode.capitalize()} mode")

    plt.suptitle("Precursor vs. Fragment m/z", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_example_spectra(df: pd.DataFrame, output_path: str, n_examples: int = 6):
    """Plot example MS2 spectra as mirror plots or stick spectra."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # Select diverse examples from different lipid classes
    top_classes = df["lipid_class"].value_counts().head(n_examples).index

    for ax, lipid_class in zip(axes, top_classes):
        # Get a representative spectrum from this class
        subset = df[df["lipid_class"] == lipid_class]
        example = subset.sample(1, random_state=42).iloc[0]

        mzs = np.array(example["mz_list"])
        intensities = np.array(example["intensity_list"])

        # Normalize intensities to 100
        if max(intensities) > 0:
            intensities = intensities / max(intensities) * 100

        # Create stick spectrum
        ax.vlines(mzs, 0, intensities, colors="#1f77b4", linewidth=1)
        ax.scatter(mzs, intensities, s=10, color="#1f77b4", zorder=5)

        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative intensity (%)")
        ax.set_title(f"{example['name'][:30]}...\n{example['adduct_name']}", fontsize=9)
        ax.set_ylim(0, 110)

        # Add precursor annotation
        ax.axvline(example["precursor_mz"], color="red", linestyle="--",
                   alpha=0.5, linewidth=1)
        ax.text(example["precursor_mz"], 105, "Precursor", fontsize=7,
                ha="center", color="red")

    plt.suptitle("Example MS2 Spectra by Lipid Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_statistics(df: pd.DataFrame, output_path: str):
    """Create a summary statistics panel."""
    fig = plt.figure(figsize=(12, 8))

    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Dataset overview (text)
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")

    stats_text = f"""Dataset Summary

Total spectra: {len(df):,}
Positive mode: {len(df[df['mode'] == 'positive']):,}
Negative mode: {len(df[df['mode'] == 'negative']):,}

Unique lipid classes: {df['lipid_class'].nunique():,}
Unique adducts: {df['adduct_name'].nunique():,}

Precursor m/z range:
  Min: {df['precursor_mz'].min():.2f}
  Max: {df['precursor_mz'].max():.2f}
  Mean: {df['precursor_mz'].mean():.2f}

Peaks per spectrum:
  Min: {df['n_peaks'].min()}
  Max: {df['n_peaks'].max()}
  Mean: {df['n_peaks'].mean():.1f}
  Median: {df['n_peaks'].median():.0f}"""

    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.3))

    # 2. Pie chart of ion modes
    ax = fig.add_subplot(gs[0, 1])
    mode_counts = df["mode"].value_counts()
    colors = ["#1f77b4", "#d62728"]
    ax.pie(mode_counts.values, labels=mode_counts.index.str.capitalize(),
           autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Ion Mode Distribution")

    # 3. Precursor m/z density by mode
    ax = fig.add_subplot(gs[0, 2])
    for mode, color in [("positive", "#1f77b4"), ("negative", "#d62728")]:
        subset = df[df["mode"] == mode]
        sns.kdeplot(data=subset["precursor_mz"], ax=ax, color=color,
                    label=mode.capitalize(), fill=True, alpha=0.3)
    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Density")
    ax.set_title("Precursor m/z Density")
    ax.legend()

    # 4. Top 10 lipid classes
    ax = fig.add_subplot(gs[1, 0])
    top_classes = df["lipid_class"].value_counts().head(10)
    ax.bar(range(len(top_classes)), top_classes.values,
           color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_classes))))
    ax.set_xticks(range(len(top_classes)))
    ax.set_xticklabels(top_classes.index, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top 10 Lipid Classes")

    # 5. Peaks per spectrum distribution
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(df["n_peaks"], bins=30, alpha=0.8, color="#2ca02c", edgecolor="white")
    ax.axvline(df["n_peaks"].median(), color="red", linestyle="--",
               label=f"Median: {df['n_peaks'].median():.0f}")
    ax.set_xlabel("Peaks per spectrum")
    ax.set_ylabel("Count")
    ax.set_title("Fragment Peak Distribution")
    ax.legend()

    # 6. Precursor m/z vs peaks scatter (sampled)
    ax = fig.add_subplot(gs[1, 2])
    sample = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(sample["precursor_mz"], sample["n_peaks"], alpha=0.2, s=5, c="#7f7f7f")
    ax.set_xlabel("Precursor m/z")
    ax.set_ylabel("Number of peaks")
    ax.set_title("m/z vs. Fragment Count")

    plt.suptitle("LipidBlast MS2 Dataset Overview", fontsize=16, fontweight="bold", y=1.02)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Run all analyses and generate figures."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_data(PARQUET_FILE)

    # Generate all figures
    print("\nGenerating figures...")

    plot_summary_statistics(df, f"{OUTPUT_DIR}/01_summary_statistics.png")
    plot_precursor_mz_distribution(df, f"{OUTPUT_DIR}/02_precursor_mz_distribution.png")
    plot_peaks_per_spectrum(df, f"{OUTPUT_DIR}/03_peaks_per_spectrum.png")
    plot_lipid_class_distribution(df, f"{OUTPUT_DIR}/04_lipid_class_distribution.png")
    plot_adduct_distribution(df, f"{OUTPUT_DIR}/05_adduct_distribution.png")
    plot_mz_vs_peaks(df, f"{OUTPUT_DIR}/06_mz_vs_peaks.png")
    plot_intensity_distribution(df, f"{OUTPUT_DIR}/07_intensity_distribution.png")
    plot_mz_heatmap(df, f"{OUTPUT_DIR}/08_precursor_fragment_heatmap.png")
    plot_example_spectra(df, f"{OUTPUT_DIR}/09_example_spectra.png")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
