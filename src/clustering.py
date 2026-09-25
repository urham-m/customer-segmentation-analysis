from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
)

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import (
    QuantileTransformer
)


ROOT = Path(__file__).resolve().parents[1]

RFM_FILE = (
    ROOT
    / "data"
    / "processed"
    / "customer_rfm.csv"
)

USERS_FILE = (
    ROOT
    / "data"
    / "raw"
    / "users.csv"
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

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEGMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURES = [
    "recency",
    "frequency",
    "monetary",
]

# These are the k values required by the assignment.
K_VALUES = [
    3,
    4,
    5,
    6,
]

# Locked final model for this dataset.
FINAL_K = 4

RANDOM_STATE = 42


# -------------------------------------------------------------
# SEGMENT NAMING
# -------------------------------------------------------------

def segment_name_map(
    profiles: pd.DataFrame
) -> dict[int, str]:

    """
    Create business-friendly names from the actual
    RFM profiles produced by K-means.
    """

    overall_recency = (
        profiles["recency_median"].median()
    )

    overall_frequency = (
        profiles["frequency_mean"].mean()
    )

    overall_monetary = (
        profiles["monetary_mean"].mean()
    )

    names = {}

    remaining = set(
        profiles.index.astype(int)
    )

    # ---------------------------------------------------------
    # CHAMPIONS
    # Recent + frequent + valuable
    # ---------------------------------------------------------

    champion_candidates = profiles[
        (profiles["recency_median"] <= overall_recency)
        &
        (profiles["frequency_mean"] >= overall_frequency)
    ]

    if not champion_candidates.empty:

        champion = int(
            champion_candidates[
                "monetary_mean"
            ].idxmax()
        )

        names[
            champion
        ] = "Champions"

        remaining.discard(
            champion
        )

    # ---------------------------------------------------------
    # AT-RISK HIGH VALUE
    # Older customers but still relatively valuable
    # ---------------------------------------------------------

    high_value_risk_candidates = profiles[
        (profiles["recency_median"] > overall_recency)
        &
        (profiles["monetary_mean"] >= overall_monetary)
    ]

    high_value_risk_candidates = (
        high_value_risk_candidates.loc[
            high_value_risk_candidates.index.intersection(
                remaining
            )
        ]
    )

    if not high_value_risk_candidates.empty:

        at_risk_high_value = int(
            high_value_risk_candidates[
                "monetary_mean"
            ].idxmax()
        )

        names[
            at_risk_high_value
        ] = "At-Risk High Value"

        remaining.discard(
            at_risk_high_value
        )

    # ---------------------------------------------------------
    # AT-RISK LOW VALUE
    # Older + relatively low monetary value
    # ---------------------------------------------------------

    low_value_risk_candidates = profiles[
        (profiles["recency_median"] > overall_recency)
        &
        (profiles["monetary_mean"] < overall_monetary)
    ]

    low_value_risk_candidates = (
        low_value_risk_candidates.loc[
            low_value_risk_candidates.index.intersection(
                remaining
            )
        ]
    )

    if not low_value_risk_candidates.empty:

        at_risk_low_value = int(
            low_value_risk_candidates[
                "monetary_mean"
            ].idxmin()
        )

        names[
            at_risk_low_value
        ] = "At-Risk Low Value"

        remaining.discard(
            at_risk_low_value
        )

    # Whatever cluster remains represents the more
    # recent/occasional customer group.
    for cluster_id in sorted(remaining):

        names[
            cluster_id
        ] = "New / Occasional"

    return names


# -------------------------------------------------------------
# MARKETING RECOMMENDATIONS
# -------------------------------------------------------------

RECOMMENDATIONS = {

    "Champions": (
        "Use loyalty/VIP benefits, early access, "
        "personalized cross-sells, and referral "
        "incentives to deepen retention and advocacy."
    ),

    "At-Risk High Value": (
        "Run personalized win-back campaigns, "
        "highlight relevant products, and use "
        "limited-time offers to recover previously "
        "valuable customers."
    ),

    "At-Risk Low Value": (
        "Use low-cost reactivation campaigns, "
        "educational content, and frequency-building "
        "offers before applying expensive incentives."
    ),

    "New / Occasional": (
        "Focus on the second purchase: onboarding "
        "messages, cross-sell recommendations, "
        "and a modest next-purchase incentive."
    ),

    "No Purchase Yet": (
        "Use welcome/onboarding messaging and a "
        "first-purchase incentive to convert "
        "registered users into buyers."
    ),
}


def main() -> None:

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    rfm = pd.read_csv(
        RFM_FILE,
        parse_dates=[
            "last_purchase_date"
        ]
    )

    users = pd.read_csv(
        USERS_FILE,
        parse_dates=[
            "signup_date"
        ]
    )

    X = rfm[
        FEATURES
    ].copy()

    # ---------------------------------------------------------
    # 2. NORMALIZE RFM FEATURES
    # ---------------------------------------------------------

    # RFM variables are skewed, especially monetary value.
    # Quantile transformation puts them on comparable scales.
    transformer = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(
            1000,
            len(X)
        ),
        random_state=RANDOM_STATE,
    )

    X_transformed = (
        transformer.fit_transform(X)
    )

    # ---------------------------------------------------------
    # 3. TEST K = 3, 4, 5, 6
    # ---------------------------------------------------------

    metric_rows = []

    for k in K_VALUES:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
        )

        labels = model.fit_predict(
            X_transformed
        )

        silhouette = silhouette_score(
            X_transformed,
            labels
        )

        metric_rows.append(
            {
                "model": "KMeans",
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette,
            }
        )

    metrics = pd.DataFrame(
        metric_rows
    )

    metrics[
        "selected_final_k"
    ] = metrics["k"].eq(
        FINAL_K
    )

    metrics.to_csv(
        METRICS_DIR
        / "cluster_metrics.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 4. FINAL K-MEANS MODEL
    # ---------------------------------------------------------

    final_kmeans = KMeans(
        n_clusters=FINAL_K,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    rfm[
        "kmeans_cluster"
    ] = final_kmeans.fit_predict(
        X_transformed
    )

    final_silhouette = silhouette_score(
        X_transformed,
        rfm["kmeans_cluster"]
    )

    # ---------------------------------------------------------
    # 5. HIERARCHICAL CLUSTERING COMPARISON
    # ---------------------------------------------------------

    hierarchical = (
        AgglomerativeClustering(
            n_clusters=FINAL_K,
            linkage="ward",
        )
    )

    rfm[
        "hierarchical_cluster"
    ] = hierarchical.fit_predict(
        X_transformed
    )

    hierarchical_silhouette = (
        silhouette_score(
            X_transformed,
            rfm[
                "hierarchical_cluster"
            ]
        )
    )

    comparison = pd.DataFrame(
        [
            {
                "model": "KMeans",
                "k": FINAL_K,
                "silhouette_score":
                    final_silhouette,
            },
            {
                "model":
                    "Agglomerative Ward",
                "k": FINAL_K,
                "silhouette_score":
                    hierarchical_silhouette,
            },
        ]
    )

    comparison.to_csv(
        METRICS_DIR
        / "hierarchical_comparison.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 6. BUILD CLUSTER PROFILES
    # ---------------------------------------------------------

    profiles = (
        rfm
        .groupby("kmeans_cluster")
        .agg(
            customer_count=(
                "user_id",
                "count"
            ),
            recency_mean=(
                "recency",
                "mean"
            ),
            recency_median=(
                "recency",
                "median"
            ),
            frequency_mean=(
                "frequency",
                "mean"
            ),
            frequency_median=(
                "frequency",
                "median"
            ),
            monetary_mean=(
                "monetary",
                "mean"
            ),
            monetary_median=(
                "monetary",
                "median"
            ),
        )
        .sort_index()
    )

    names = segment_name_map(
        profiles
    )

    profiles[
        "segment_name"
    ] = profiles.index.map(
        names
    )

    profiles[
        "marketing_recommendation"
    ] = profiles[
        "segment_name"
    ].map(
        RECOMMENDATIONS
    )

    profiles.to_csv(
        METRICS_DIR
        / "cluster_profiles.csv"
    )

    # ---------------------------------------------------------
    # 7. ADD SEGMENT NAMES TO BUYERS
    # ---------------------------------------------------------

    rfm[
        "segment_name"
    ] = rfm[
        "kmeans_cluster"
    ].map(
        names
    )

    rfm[
        "marketing_recommendation"
    ] = rfm[
        "segment_name"
    ].map(
        RECOMMENDATIONS
    )

    buyer_segments = rfm[
        [
            "user_id",
            "recency",
            "frequency",
            "monetary",
            "last_purchase_date",
            "kmeans_cluster",
            "hierarchical_cluster",
            "segment_name",
            "marketing_recommendation",
        ]
    ].copy()

    # ---------------------------------------------------------
    # 8. MERGE BACK TO ALL REGISTERED USERS
    # ---------------------------------------------------------

    final = (
        users[
            [
                "user_id",
                "name",
                "gender",
                "city",
                "signup_date",
            ]
        ]
        .merge(
            buyer_segments,
            on="user_id",
            how="left"
        )
    )

    # ---------------------------------------------------------
    # 9. HANDLE USERS WITH NO PURCHASE
    # ---------------------------------------------------------

    no_purchase = (
        final["kmeans_cluster"].isna()
    )

    final.loc[
        no_purchase,
        "frequency"
    ] = 0

    final.loc[
        no_purchase,
        "monetary"
    ] = 0

    final.loc[
        no_purchase,
        "segment_name"
    ] = "No Purchase Yet"

    final.loc[
        no_purchase,
        "marketing_recommendation"
    ] = RECOMMENDATIONS[
        "No Purchase Yet"
    ]

    final[
        "kmeans_cluster"
    ] = final[
        "kmeans_cluster"
    ].astype("Int64")

    final[
        "hierarchical_cluster"
    ] = final[
        "hierarchical_cluster"
    ].astype("Int64")

    final[
        "monetary"
    ] = final[
        "monetary"
    ].round(2)

    # ---------------------------------------------------------
    # 10. SAVE FINAL CUSTOMER ASSIGNMENTS
    # ---------------------------------------------------------

    final.to_csv(
        SEGMENTS_DIR
        / "customer_segments.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 11. SAVE SEGMENT SUMMARY
    # ---------------------------------------------------------

    segment_summary = (
        final
        .groupby(
            "segment_name",
            dropna=False
        )
        .agg(
            customer_count=(
                "user_id",
                "count"
            ),
            avg_recency=(
                "recency",
                "mean"
            ),
            avg_frequency=(
                "frequency",
                "mean"
            ),
            avg_monetary=(
                "monetary",
                "mean"
            ),
        )
        .reset_index()
    )

    segment_summary[
        "customer_pct"
    ] = (
        segment_summary[
            "customer_count"
        ]
        / len(final)
        * 100
    ).round(2)

    segment_summary[
        "marketing_recommendation"
    ] = segment_summary[
        "segment_name"
    ].map(
        RECOMMENDATIONS
    )

    segment_summary[
        "avg_recency"
    ] = segment_summary[
        "avg_recency"
    ].round(1)

    segment_summary[
        "avg_frequency"
    ] = segment_summary[
        "avg_frequency"
    ].round(2)

    segment_summary[
        "avg_monetary"
    ] = segment_summary[
        "avg_monetary"
    ].round(2)

    segment_summary.to_csv(
        SEGMENTS_DIR
        / "segment_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 12. PRINT RESULTS
    # ---------------------------------------------------------

    print()
    print("K-means evaluation:")
    print(
        metrics[
            [
                "k",
                "inertia",
                "silhouette_score",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Locked final k: {FINAL_K}"
    )

    print(
        f"K-means silhouette at "
        f"k={FINAL_K}: "
        f"{final_silhouette:.3f}"
    )

    print(
        "Hierarchical Ward silhouette "
        f"at k={FINAL_K}: "
        f"{hierarchical_silhouette:.3f}"
    )

    print()
    print("Segment names:")

    print(
        profiles[
            "segment_name"
        ].to_string()
    )

    print()
    print(
        "Saved customer_segments.csv "
        "and segment_summary.csv"
    )


if __name__ == "__main__":
    main()