from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
)

from sklearn.preprocessing import (
    QuantileTransformer,
)


ROOT = Path(__file__).resolve().parents[1]

RFM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "customer_rfm.csv"
)

METRICS_DIR = (
    ROOT
    / "outputs"
    / "metrics"
)

SEGMENTS_DIR = (
    ROOT
    / "outputs"
    / "segments"
)

PLOTS_DIR = (
    ROOT
    / "plots"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURES = [
    "recency",
    "frequency",
    "monetary",
]

FINAL_K = 4

RANDOM_STATE = 42


def save_plot(filename: str) -> None:

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:

    # ---------------------------------------------------------
    # LOAD OUTPUTS
    # ---------------------------------------------------------

    metrics = pd.read_csv(
        METRICS_DIR
        / "cluster_metrics.csv"
    )

    comparison = pd.read_csv(
        METRICS_DIR
        / "hierarchical_comparison.csv"
    )

    profiles = pd.read_csv(
        METRICS_DIR
        / "cluster_profiles.csv",
        index_col="kmeans_cluster",
    )

    segments = pd.read_csv(
        SEGMENTS_DIR
        / "customer_segments.csv",
        parse_dates=[
            "signup_date",
            "last_purchase_date",
        ],
    )

    rfm = pd.read_csv(
        RFM_FILE,
        parse_dates=[
            "last_purchase_date",
        ],
    )

    sns.set_theme()

    # ---------------------------------------------------------
    # 1. ELBOW CURVE
    # ---------------------------------------------------------

    elbow = (
        metrics[
            metrics["model"] == "KMeans"
        ]
        .sort_values("k")
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        elbow["k"],
        elbow["inertia"],
        marker="o",
    )

    plt.axvline(
        FINAL_K,
        linestyle="--",
        label=f"Selected k={FINAL_K}",
    )

    plt.xlabel(
        "Number of clusters (k)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        "K-means Elbow Curve"
    )

    plt.legend()

    save_plot(
        "01_elbow_curve.png"
    )

    # ---------------------------------------------------------
    # 2. SILHOUETTE SCORE BY K
    # ---------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        elbow["k"],
        elbow["silhouette_score"],
        marker="o",
    )

    # 0.45 is the lower end of the assignment's
    # ±10% tolerance around 0.50.
    plt.axhline(
        0.45,
        linestyle="--",
        label="0.45 tolerance threshold",
    )

    plt.axvline(
        FINAL_K,
        linestyle="--",
        label=f"Selected k={FINAL_K}",
    )

    plt.xlabel(
        "Number of clusters (k)"
    )

    plt.ylabel(
        "Silhouette score"
    )

    plt.title(
        "Silhouette Score by Number of Clusters"
    )

    plt.legend()

    save_plot(
        "02_silhouette_by_k.png"
    )

    # ---------------------------------------------------------
    # 3. SEGMENT DISTRIBUTION
    # ---------------------------------------------------------

    distribution = (
        segments[
            "segment_name"
        ]
        .value_counts()
        .sort_values(
            ascending=True
        )
    )

    plt.figure(
        figsize=(8, 5)
    )

    distribution.plot(
        kind="barh"
    )

    plt.xlabel(
        "Customers"
    )

    plt.ylabel(
        "Segment"
    )

    plt.title(
        "Customer Distribution by Segment"
    )

    save_plot(
        "03_segment_distribution.png"
    )

    # ---------------------------------------------------------
    # 4. RECENCY VS MONETARY VALUE
    # ---------------------------------------------------------

    buyer_segments = (
        segments[
            segments[
                "kmeans_cluster"
            ].notna()
        ].copy()
    )

    plt.figure(
        figsize=(9, 6)
    )

    sns.scatterplot(
        data=buyer_segments,
        x="recency",
        y="monetary",
        hue="segment_name",
        alpha=0.65,
        s=45,
    )

    # Monetary value is highly skewed.
    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Recency (days; lower is more recent)"
    )

    plt.ylabel(
        "Monetary value (log scale)"
    )

    plt.title(
        "RFM Customer Segments: "
        "Recency vs Monetary Value"
    )

    plt.legend(
        title="Segment",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    save_plot(
        "04_recency_vs_monetary.png"
    )

    # ---------------------------------------------------------
    # 5. CLUSTER PROFILE HEATMAP
    # ---------------------------------------------------------

    profile_cols = [
        "recency_median",
        "frequency_median",
        "monetary_median",
    ]

    profile_plot = profiles[
        profile_cols
    ].copy()

    profile_plot.index = (
        profiles[
            "segment_name"
        ]
    )

    profile_plot.columns = [
        "Recency",
        "Frequency",
        "Monetary",
    ]

    # Log transform makes monetary values easier to compare.
    profile_log = np.log1p(
        profile_plot
    )

    profile_z = (
        profile_log
        - profile_log.mean()
    ) / profile_log.std(
        ddof=0
    ).replace(
        0,
        1
    )

    plt.figure(
        figsize=(8, 5)
    )

    sns.heatmap(
        profile_z,
        annot=True,
        fmt=".2f",
        center=0,
        cmap="vlag",
    )

    plt.title(
        "Relative RFM Profile "
        "of K-means Segments"
    )

    save_plot(
        "05_cluster_profiles.png"
    )

    # ---------------------------------------------------------
    # 6. HIERARCHICAL DENDROGRAM
    # ---------------------------------------------------------

    buyer_rfm = rfm[
        FEATURES
    ].copy()

    # A sample keeps the dendrogram readable.
    sample_size = min(
        120,
        len(buyer_rfm)
    )

    sample = buyer_rfm.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    transformer = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(
            1000,
            len(buyer_rfm)
        ),
        random_state=RANDOM_STATE,
    )

    transformer.fit(
        buyer_rfm
    )

    transformed_sample = transformer.transform(
        sample
    )

    hierarchy_linkage = linkage(
        transformed_sample,
        method="ward",
    )

    plt.figure(
        figsize=(12, 6)
    )

    dendrogram(
        hierarchy_linkage,
        no_labels=True,
        color_threshold=None,
    )

    plt.xlabel(
        "Sampled customers"
    )

    plt.ylabel(
        "Ward linkage distance"
    )

    plt.title(
        "Hierarchical Clustering Dendrogram "
        f"(sample of {sample_size} buyers)"
    )

    save_plot(
        "06_hierarchical_dendrogram.png"
    )

    # ---------------------------------------------------------
    # 7. MODEL COMPARISON
    # ---------------------------------------------------------

    comparison_plot = (
        comparison.copy()
    )

    comparison_plot[
        "label"
    ] = (
        comparison_plot["model"]
        + " (k="
        + comparison_plot["k"].astype(str)
        + ")"
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        comparison_plot["label"],
        comparison_plot["silhouette_score"],
    )

    plt.ylim(
        0,
        0.7
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Silhouette score"
    )

    plt.title(
        "K-means vs Hierarchical Clustering"
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    save_plot(
        "07_model_comparison.png"
    )

    print()
    print(
        f"Saved visualization files to: "
        f"{PLOTS_DIR}"
    )


if __name__ == "__main__":
    main()