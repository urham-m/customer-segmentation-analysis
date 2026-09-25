from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
METRICS_DIR = ROOT / "outputs" / "metrics"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


USERS_FILE = RAW_DIR / "users.csv"
ORDERS_FILE = RAW_DIR / "orders.csv"


REQUIRED_USER_COLUMNS = {
    "user_id",
    "name",
    "email",
    "gender",
    "city",
    "signup_date",
}

REQUIRED_ORDER_COLUMNS = {
    "order_id",
    "user_id",
    "order_date",
    "order_status",
    "total_amount",
}

# Only fulfilled purchases will be used in the RFM analysis.
FULFILLED_STATUSES = {"completed", "shipped"}


def validate_columns(df: pd.DataFrame, required: set, filename: str) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {sorted(missing)}"
        )


def main() -> None:

    print("Loading data...")

    users = pd.read_csv(USERS_FILE)
    orders = pd.read_csv(ORDERS_FILE)

    validate_columns(
        users,
        REQUIRED_USER_COLUMNS,
        "users.csv"
    )

    validate_columns(
        orders,
        REQUIRED_ORDER_COLUMNS,
        "orders.csv"
    )

    users_input_rows = len(users)
    orders_input_rows = len(orders)

    # ---------------------------------------------------------
    # 1. BASIC DATA CLEANING
    # ---------------------------------------------------------

    users["user_id"] = (
        users["user_id"]
        .astype("string")
        .str.strip()
    )

    users["signup_date"] = pd.to_datetime(
        users["signup_date"],
        errors="coerce"
    )

    users["gender"] = (
        users["gender"]
        .fillna("Unknown")
        .astype("string")
        .str.strip()
    )

    users["city"] = (
        users["city"]
        .fillna("Unknown")
        .astype("string")
        .str.strip()
    )

    orders["order_id"] = (
        orders["order_id"]
        .astype("string")
        .str.strip()
    )

    orders["user_id"] = (
        orders["user_id"]
        .astype("string")
        .str.strip()
    )

    orders["order_date"] = pd.to_datetime(
        orders["order_date"],
        errors="coerce"
    )

    orders["order_status"] = (
        orders["order_status"]
        .fillna("unknown")
        .astype("string")
        .str.strip()
        .str.lower()
    )

    orders["total_amount"] = pd.to_numeric(
        orders["total_amount"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # 2. REMOVE INVALID RECORDS
    # ---------------------------------------------------------

    users = users.dropna(
        subset=["user_id", "signup_date"]
    )

    users = users.drop_duplicates(
        subset=["user_id"]
    ).copy()

    orders = orders.dropna(
        subset=[
            "order_id",
            "user_id",
            "order_date",
            "total_amount",
        ]
    )

    orders = orders.drop_duplicates(
        subset=["order_id"]
    ).copy()

    # Purchases must have a positive monetary value.
    orders = orders[
        orders["total_amount"] > 0
    ].copy()

    # Every order should map to a known user.
    unknown_user_orders = ~orders["user_id"].isin(
        users["user_id"]
    )

    if unknown_user_orders.any():

        bad_count = int(
            unknown_user_orders.sum()
        )

        raise ValueError(
            f"Found {bad_count} orders whose user_id "
            "does not exist in users.csv."
        )

    # ---------------------------------------------------------
    # 3. KEEP ONLY FULFILLED PURCHASES
    # ---------------------------------------------------------

    fulfilled_orders = orders[
        orders["order_status"].isin(
            FULFILLED_STATUSES
        )
    ].copy()

    if fulfilled_orders.empty:
        raise ValueError(
            "No completed/shipped orders remain after cleaning."
        )

    # ---------------------------------------------------------
    # 4. DEFINE RFM SNAPSHOT DATE
    # ---------------------------------------------------------

    # One day after the latest fulfilled purchase.
    # This makes the analysis reproducible.
    snapshot_date = (
        fulfilled_orders["order_date"]
        .dt.normalize()
        .max()
        + pd.Timedelta(days=1)
    )

    # ---------------------------------------------------------
    # 5. CALCULATE RFM
    # ---------------------------------------------------------

    rfm = (
        fulfilled_orders
        .groupby("user_id")
        .agg(
            last_purchase_date=(
                "order_date",
                "max"
            ),
            frequency=(
                "order_id",
                "nunique"
            ),
            monetary=(
                "total_amount",
                "sum"
            ),
        )
        .reset_index()
    )

    rfm["recency"] = (
        snapshot_date
        - rfm["last_purchase_date"].dt.normalize()
    ).dt.days

    rfm = rfm[
        [
            "user_id",
            "recency",
            "frequency",
            "monetary",
            "last_purchase_date",
        ]
    ].copy()

    rfm["monetary"] = rfm["monetary"].round(2)

    # ---------------------------------------------------------
    # 6. SAVE RFM DATASET
    # ---------------------------------------------------------

    rfm_file = (
        PROCESSED_DIR
        / "customer_rfm.csv"
    )

    rfm.to_csv(
        rfm_file,
        index=False
    )

    # ---------------------------------------------------------
    # 7. SAVE DATA QUALITY SUMMARY
    # ---------------------------------------------------------

    quality_summary = pd.DataFrame(
        [
            [
                "users_input_rows",
                users_input_rows
            ],
            [
                "users_after_cleaning",
                len(users)
            ],
            [
                "orders_input_rows",
                orders_input_rows
            ],
            [
                "orders_after_cleaning",
                len(orders)
            ],
            [
                "fulfilled_orders_used_for_rfm",
                len(fulfilled_orders)
            ],
            [
                "customers_with_fulfilled_orders",
                rfm["user_id"].nunique()
            ],
            [
                "customers_without_fulfilled_orders",
                (
                    users["user_id"].nunique()
                    - rfm["user_id"].nunique()
                )
            ],
            [
                "snapshot_date",
                snapshot_date.date().isoformat()
            ],
        ],
        columns=[
            "metric",
            "value"
        ]
    )

    quality_summary.to_csv(
        METRICS_DIR
        / "data_quality_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. SAVE ORDER STATUS COUNTS
    # ---------------------------------------------------------

    status_counts = (
        orders["order_status"]
        .value_counts()
        .rename_axis("order_status")
        .reset_index(name="order_count")
    )

    status_counts.to_csv(
        METRICS_DIR
        / "order_status_counts.csv",
        index=False
    )

    print()
    print("Preprocessing complete.")
    print(f"RFM file: {rfm_file}")
    print(
        f"Customers used for clustering: {len(rfm):,}"
    )
    print(
        f"RFM snapshot date: "
        f"{snapshot_date.date()}"
    )


if __name__ == "__main__":
    main()