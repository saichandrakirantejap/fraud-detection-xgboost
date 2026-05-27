"""
pipeline.py

Full PySpark feature pipeline for fraud detection.

Features are grouped into four families:
1. Velocity     — how many transactions in recent windows
2. Rolling      — aggregate stats over time windows
3. Merchant     — category-level behavioral patterns
4. Device       — consistency of device fingerprint

The null merchant category observation was an early EDA finding —
null merchant codes appeared more often in fraudulent transactions.
So missingness itself became a feature.
"""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def build_features(df: DataFrame) -> DataFrame:
    """
    Run the full feature pipeline on raw transaction data.

    Input columns expected:
        transaction_id, customer_id, amount, merchant_category,
        merchant_id, device_id, timestamp
    """
    df = add_velocity_features(df)
    df = add_rolling_features(df)
    df = add_merchant_features(df)
    df = add_device_features(df)
    return df


def add_velocity_features(df: DataFrame) -> DataFrame:
    """
    Transaction velocity — how many transactions has this customer
    made in the last N hours.
    """
    window_1h = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-3600, 0)           # last 1 hour in seconds
    )
    window_24h = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-86400, 0)          # last 24 hours
    )

    df = df.withColumn("txn_count_1h",  F.count("transaction_id").over(window_1h))
    df = df.withColumn("txn_count_24h", F.count("transaction_id").over(window_24h))
    return df


def add_rolling_features(df: DataFrame) -> DataFrame:
    """
    Rolling spend averages over 7 and 30 day windows.
    A transaction amount that is much higher than the customer's
    average is a strong fraud signal.
    """
    window_7d = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-7 * 86400, 0)
    )
    window_30d = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-30 * 86400, 0)
    )

    df = df.withColumn("avg_amount_7d",  F.avg("amount").over(window_7d))
    df = df.withColumn("avg_amount_30d", F.avg("amount").over(window_30d))
    df = df.withColumn("max_amount_30d", F.max("amount").over(window_30d))

    # Ratio of current amount to rolling average — high ratios flag anomalies
    df = df.withColumn(
        "amount_to_avg_7d_ratio",
        F.when(F.col("avg_amount_7d") > 0, F.col("amount") / F.col("avg_amount_7d"))
        .otherwise(0.0)
    )

    return df


def add_merchant_features(df: DataFrame) -> DataFrame:
    """
    Merchant category features.

    Null merchant category was found in EDA to correlate with fraud —
    so we treat missingness as a feature rather than imputing it away.
    """
    df = df.withColumn(
        "merchant_category_is_null",
        F.when(F.col("merchant_category").isNull(), 1).otherwise(0)
    )
    df = df.withColumn(
        "merchant_category_filled",
        F.coalesce(F.col("merchant_category"), F.lit("UNKNOWN"))
    )

    # How many times has this customer transacted at this merchant?
    customer_merchant_window = Window.partitionBy("customer_id", "merchant_id")
    df = df.withColumn(
        "customer_merchant_txn_count",
        F.count("transaction_id").over(customer_merchant_window)
    )

    return df


def add_device_features(df: DataFrame) -> DataFrame:
    """
    Device consistency — has this customer used this device before?
    A new device on a high-value transaction is suspicious.
    """
    customer_device_window = Window.partitionBy("customer_id", "device_id")
    df = df.withColumn(
        "device_seen_before",
        F.when(
            F.count("transaction_id").over(customer_device_window) > 1, 1
        ).otherwise(0)
    )
    return df
