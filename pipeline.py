"""Data preparation and clustering for the customer behavior dashboard.

The default dataset is synthetic and intended for demonstration only. Real
REES46 results require uploading the source CSV. Large-file ETL can use the
optional Spark local[*] path; model fitting and visualization remain local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = {
    "event_time",
    "event_type",
    "product_id",
    "price",
    "user_id",
    "user_session",
}
BASELINE_COLS = ["Recency", "Frequency", "Monetary"]
PROPOSED_COLS = [
    *BASELINE_COLS,
    "Purchase_Flag",
    "View_Count",
    "Cart_Count",
    "Cart_to_Purchase_Gap",
    "Interaction_Span",
    "Total_Sessions",
]


def process_csv_with_spark(csv_bytes: bytes) -> dict:
    """Clean and aggregate an uploaded CSV with Spark local[*].

    PySpark and a compatible Java runtime are optional. Clustering remains in
    scikit-learn after customer-level aggregation, matching the study's split
    between large-event ETL and sampled DBSCAN evaluation.
    """
    try:
        from pyspark.sql import SparkSession, functions as F
        from pyspark import StorageLevel
    except ImportError as exc:
        raise ValueError(
            "Chế độ Spark cần cài PySpark và Java. Cài theo requirements-spark.txt, "
            "hoặc chọn chế độ Pandas cho tệp nhỏ hơn."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="customer_events_") as temp_dir:
        csv_path = Path(temp_dir) / "events.csv"
        csv_path.write_bytes(csv_bytes)
        try:
            spark = (
                SparkSession.builder
                .master("local[*]")
                .appName("EcommerceCustomerBehaviorDashboard")
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
        except Exception as exc:
            raise ValueError(
                "Không khởi tạo được Spark local[*]. Hãy kiểm tra PySpark, Java tương thích "
                "và cấu hình JAVA_HOME; hoặc chuyển sang Pandas với tệp nhỏ hơn."
            ) from exc
        spark.sparkContext.setLogLevel("ERROR")
        try:
            raw = spark.read.option("header", True).option("mode", "PERMISSIVE").csv(str(csv_path))
            raw = raw.toDF(*[str(column).strip() for column in raw.columns])
            missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
            if missing:
                raise ValueError(
                    "Thiếu cột bắt buộc: " + ", ".join(missing)
                    + ". Cần event_time, event_type, product_id, price, user_id và user_session."
                )

            typed = raw.select(
                F.to_timestamp(
                    F.regexp_replace(F.trim(F.col("event_time")), r"\s+UTC$", "")
                ).alias("event_time"),
                F.lower(F.trim(F.col("event_type"))).alias("event_type"),
                F.col("product_id").cast("string").alias("product_id"),
                F.col("price").cast("double").alias("price"),
                F.col("user_id").cast("string").alias("user_id"),
                F.col("user_session").cast("string").alias("user_session"),
            )
            raw_count = int(typed.count())
            initial_users = int(
                typed.where(F.col("user_id").isNotNull()).select("user_id").distinct().count()
            )

            valid = typed.where(
                F.col("event_time").isNotNull()
                & F.col("event_type").isNotNull()
                & (F.length(F.col("event_type")) > 0)
                & F.col("product_id").isNotNull()
                & F.col("user_id").isNotNull()
                & F.col("user_session").isNotNull()
                & (F.col("price") > 0)
            )
            valid = valid.persist(StorageLevel.DISK_ONLY)
            valid_count = int(valid.count())
            valid_users = int(valid.select("user_id").distinct().count())

            deduped = valid.dropDuplicates(["event_time", "user_id", "product_id", "event_type"])
            deduped = deduped.persist(StorageLevel.DISK_ONLY)
            deduped_count = int(deduped.count())
            deduped_users = int(deduped.select("user_id").distinct().count())

            eligible = (
                deduped.groupBy("user_id")
                .count()
                .where(F.col("count") >= 2)
                .select("user_id")
            )
            clean = deduped.join(eligible, on="user_id", how="inner").persist(StorageLevel.DISK_ONLY)
            clean_count = int(clean.count())

            event_counts = (
                clean.groupBy("event_type")
                .count()
                .orderBy(F.desc("count"))
                .toPandas()
                .rename(columns={"event_type": "Loại sự kiện", "count": "Số sự kiện"})
            )
            t_ref_row = clean.agg(F.max("event_time").alias("t_ref")).first()
            t_ref = t_ref_row["t_ref"]
            if t_ref is None:
                raise ValueError("Không có thời điểm hợp lệ sau khi làm sạch dữ liệu.")

            agg = clean.groupBy("user_id").agg(
                F.max("event_time").alias("Last_Interaction"),
                F.min("event_time").alias("First_Interaction"),
                F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("Frequency"),
                F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0.0)).alias("Monetary"),
                F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("View_Count"),
                F.sum(F.when(F.col("event_type") == "cart", 1).otherwise(0)).alias("Cart_Count"),
                F.countDistinct("user_session").alias("Total_Sessions"),
                F.max(F.when(F.col("event_type") == "purchase", F.col("event_time"))).alias("Last_Purchase"),
            )
            anchor = F.coalesce(F.col("Last_Purchase"), F.col("Last_Interaction"))
            features_spark = agg.select(
                "user_id",
                ((F.unix_timestamp(F.lit(t_ref)) - F.unix_timestamp(anchor)) / 86400.0).alias("Recency"),
                F.col("Frequency"),
                F.col("Monetary"),
                (F.col("Frequency") > 0).cast("int").alias("Purchase_Flag"),
                F.col("View_Count"),
                F.col("Cart_Count"),
                F.when(
                    F.col("Cart_Count") > 0,
                    1.0 - F.least(F.lit(1.0), F.col("Frequency") / F.col("Cart_Count")),
                ).otherwise(0.0).alias("Cart_to_Purchase_Gap"),
                ((F.unix_timestamp(F.col("Last_Interaction")) - F.unix_timestamp(F.col("First_Interaction"))) / 86400.0).alias("Interaction_Span"),
                F.col("Total_Sessions"),
            )
            features = features_spark.toPandas()
            features[PROPOSED_COLS] = features[PROPOSED_COLS].fillna(0)

            funnel = pd.DataFrame([
                {"Giai đoạn": "Dữ liệu đầu vào", "Số sự kiện": raw_count, "Số khách hàng": initial_users,
                 "Tỷ lệ khách giữ lại": 1.0 if initial_users else 0.0},
                {"Giai đoạn": "Loại sự kiện thiếu/lỗi", "Số sự kiện": valid_count, "Số khách hàng": valid_users,
                 "Tỷ lệ khách giữ lại": valid_users / initial_users if initial_users else 0.0},
                {"Giai đoạn": "Khử trùng lặp", "Số sự kiện": deduped_count, "Số khách hàng": deduped_users,
                 "Tỷ lệ khách giữ lại": deduped_users / initial_users if initial_users else 0.0},
                {"Giai đoạn": "Giữ khách có từ 2 sự kiện", "Số sự kiện": clean_count, "Số khách hàng": len(features),
                 "Tỷ lệ khách giữ lại": len(features) / initial_users if initial_users else 0.0},
            ])
            period = clean.agg(
                F.min("event_time").alias("start"), F.max("event_time").alias("end")
            ).first()
            return {
                "features": features[["user_id", *PROPOSED_COLS]],
                "funnel": funnel,
                "event_counts": event_counts,
                "raw_event_count": raw_count,
                "clean_event_count": clean_count,
                "period_start": period["start"],
                "period_end": period["end"],
            }
        finally:
            spark.stop()


def validate_raw_data(df: pd.DataFrame) -> None:
    """Raise a readable error when a CSV cannot support the research features."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Tệp dữ liệu đang trống hoặc không phải bảng CSV hợp lệ.")
    available_columns = {str(column).strip() for column in df.columns}
    missing = sorted(REQUIRED_COLUMNS - available_columns)
    if missing:
        raise ValueError(
            "Thiếu cột bắt buộc: " + ", ".join(missing)
            + ". Cần tối thiểu event_time, event_type, product_id, price, "
            "user_id và user_session."
        )


def generate_mock_rees46_data(n_events: int = 50_000, n_users: int = 2_000) -> pd.DataFrame:
    """Generate behavior-shaped synthetic data for UI demonstration only.

    Event mix depends on a user's synthetic cohort so that the dashboard has
    visible patterns. It is not sampled from REES46 and must not be presented
    as a reproduction of the study's empirical results.
    """
    if n_events < 100 or n_users < 10:
        raise ValueError("Dữ liệu demo cần ít nhất 100 sự kiện và 10 khách hàng.")

    rng = np.random.default_rng(42)
    duplicate_count = max(1, int(n_events * 0.03))
    base_count = n_events - duplicate_count
    single_count = min(int(n_users * 0.15), base_count // 3)
    core_count = max(1, n_users - single_count)

    core_users = np.array([f"demo_{i:05d}" for i in range(core_count)])
    one_event_users = np.array(
        [f"demo_{i:05d}" for i in range(core_count, core_count + single_count)]
    )
    users = np.concatenate(
        [rng.choice(core_users, size=base_count - single_count), one_event_users]
    )
    rng.shuffle(users)

    # Four illustrative cohorts: repeat/high value, browsers, cart abandoners,
    # and occasional buyers. The cohort is only a synthetic data-generating aid.
    cohort_by_user = {
        user: min(3, (idx * 4) // max(1, core_count))
        for idx, user in enumerate(core_users)
    }
    event_types = np.empty(base_count, dtype=object)
    event_options = np.array(["view", "cart", "remove_from_cart", "purchase"])
    cohort_probabilities = np.array([
        [0.48, 0.20, 0.10, 0.22],
        [0.85, 0.08, 0.06, 0.01],
        [0.54, 0.34, 0.10, 0.02],
        [0.61, 0.18, 0.13, 0.08],
    ])
    cohort_ids = np.array([cohort_by_user.get(user, 1) for user in users])
    for cohort_id, probabilities in enumerate(cohort_probabilities):
        mask = cohort_ids == cohort_id
        event_types[mask] = rng.choice(event_options, size=int(mask.sum()), p=probabilities)

    start = np.datetime64("2019-10-01T00:00:00")
    end = np.datetime64("2020-02-29T00:00:00")
    timestamp_span = int((end - start) / np.timedelta64(1, "s"))
    timestamps = start + rng.integers(0, timestamp_span, size=base_count).astype("timedelta64[s]")
    df = pd.DataFrame({
        "event_time": pd.to_datetime(timestamps),
        "event_type": event_types,
        "product_id": rng.integers(1000, 5000, size=base_count),
        "category_id": rng.integers(10, 50, size=base_count),
        "brand": rng.choice(
            ["loreal", "maybelline", "innisfree", "mac", "unknown"],
            size=base_count,
        ),
        "price": np.round(rng.lognormal(mean=2.6, sigma=0.75, size=base_count), 2),
        "user_id": users,
        "user_session": [f"sess_{rng.integers(1, 12_000)}" for _ in range(base_count)],
    })

    # Add controlled data-quality examples so the data funnel can be explored.
    df = pd.concat([df, df.iloc[:duplicate_count].copy()], ignore_index=True)
    bad_price_count = min(int(n_events * 0.02), len(df))
    bad_price_idx = rng.choice(df.index, size=bad_price_count, replace=False)
    df.loc[bad_price_idx, "price"] = -1.0

    eligible = df.index[df["user_id"].isin(core_users[: min(50, core_count)])].to_numpy()
    null_count = min(int(n_events * 0.015), len(eligible))
    if null_count:
        null_idx = rng.choice(eligible, size=null_count, replace=False)
        df.loc[null_idx, "user_id"] = np.nan
    return df


def _normalise_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work.columns = [str(column).strip() for column in work.columns]
    validate_raw_data(work)
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce", utc=True)
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["event_type"] = work["event_type"].astype("string").str.strip().str.lower()
    return work


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean event rows, de-duplicate, and retain users with >=2 events."""
    work = _normalise_raw_data(df)
    work = work.dropna(subset=["event_time", "event_type", "product_id", "user_id", "user_session"])
    work = work.loc[work["price"] > 0].copy()
    work = work.drop_duplicates(
        subset=["event_time", "user_id", "product_id", "event_type"]
    )
    user_counts = work["user_id"].value_counts()
    valid_users = user_counts.index[user_counts >= 2]
    return work.loc[work["user_id"].isin(valid_users)].copy()


def extract_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction and clickstream features at customer level."""
    if df_clean.empty:
        raise ValueError("Không còn sự kiện hợp lệ sau bước làm sạch dữ liệu.")

    work = df_clean.copy()
    work["is_purchase"] = work["event_type"].eq("purchase").astype("int8")
    work["is_view"] = work["event_type"].eq("view").astype("int8")
    work["is_cart"] = work["event_type"].eq("cart").astype("int8")
    work["purchase_price"] = work["price"].where(work["is_purchase"].eq(1), 0.0)
    t_ref = work["event_time"].max()

    grouped = work.groupby("user_id", sort=False)
    features = grouped.agg(
        Last_Interaction=("event_time", "max"),
        First_Interaction=("event_time", "min"),
        Frequency=("is_purchase", "sum"),
        Monetary=("purchase_price", "sum"),
        View_Count=("is_view", "sum"),
        Cart_Count=("is_cart", "sum"),
        Total_Sessions=("user_session", "nunique"),
    )
    last_purchase = work.loc[work["is_purchase"].eq(1)].groupby("user_id")["event_time"].max()
    features["Last_Purchase"] = last_purchase
    features["Purchase_Flag"] = features["Frequency"].gt(0).astype("int8")

    # Match the study's adjusted R: purchase recency for buyers; last interaction
    # recency for customers who have not yet purchased (no fixed penalty).
    recency_anchor = features["Last_Purchase"].fillna(features["Last_Interaction"])
    features["Recency"] = (t_ref - recency_anchor).dt.total_seconds() / 86_400
    features["Interaction_Span"] = (
        features["Last_Interaction"] - features["First_Interaction"]
    ).dt.total_seconds() / 86_400

    # This is the paper's aggregate cart-to-purchase proxy, not an order-level
    # conversion rate because event logs do not link a cart event to an order.
    cart_counts = features["Cart_Count"].to_numpy(dtype=float)
    purchase_counts = features["Frequency"].to_numpy(dtype=float)
    gap = np.zeros(len(features), dtype=float)
    has_cart = cart_counts > 0
    gap[has_cart] = 1.0 - np.minimum(1.0, purchase_counts[has_cart] / cart_counts[has_cart])
    features["Cart_to_Purchase_Gap"] = gap

    features = features.reset_index()
    numeric_cols = PROPOSED_COLS
    features[numeric_cols] = features[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    return features[["user_id", *PROPOSED_COLS]]


def prepare_datasets(features_df: pd.DataFrame):
    """Apply log1p to skewed non-negative fields and standardise both designs."""
    if len(features_df) < 3:
        raise ValueError("Cần tối thiểu 3 khách hàng hợp lệ để phân cụm.")

    transformed = features_df.copy()
    log_columns = [
        "Recency", "Frequency", "Monetary", "View_Count", "Cart_Count",
        "Interaction_Span", "Total_Sessions",
    ]
    for column in log_columns:
        transformed[column] = np.log1p(transformed[column].clip(lower=0))

    base_scaler = StandardScaler()
    proposed_scaler = StandardScaler()
    x_base = base_scaler.fit_transform(transformed[BASELINE_COLS])
    x_proposed = proposed_scaler.fit_transform(transformed[PROPOSED_COLS])
    return x_base, x_proposed, BASELINE_COLS.copy(), PROPOSED_COLS.copy()


def get_data_funnel_stats(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Count rows and users at each cleaning stage without mutating the input."""
    raw = df_raw.copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    validate_raw_data(raw)
    raw["event_time"] = pd.to_datetime(raw["event_time"], errors="coerce", utc=True)
    raw["price"] = pd.to_numeric(raw["price"], errors="coerce")
    raw["event_type"] = raw["event_type"].astype("string").str.strip()
    initial_users = int(raw["user_id"].nunique())
    stats = []

    def add_stage(name: str, current: pd.DataFrame):
        users = int(current["user_id"].nunique())
        stats.append({
            "Giai đoạn": name,
            "Số sự kiện": int(len(current)),
            "Số khách hàng": users,
            "Tỷ lệ khách giữ lại": users / initial_users if initial_users else 0.0,
        })

    add_stage("Dữ liệu đầu vào", raw)
    valid = raw.dropna(subset=[
        "event_time", "event_type", "product_id", "user_id", "user_session"
    ])
    valid = valid.loc[valid["price"] > 0].copy()
    add_stage("Loại sự kiện thiếu/lỗi", valid)
    deduped = valid.drop_duplicates(
        subset=["event_time", "user_id", "product_id", "event_type"]
    )
    add_stage("Khử trùng lặp", deduped)
    counts = deduped["user_id"].value_counts()
    retained_users = counts.index[counts >= 2]
    retained = deduped.loc[deduped["user_id"].isin(retained_users)]
    add_stage("Giữ khách có từ 2 sự kiện", retained)
    return pd.DataFrame(stats)


def _stratified_indices(strata: pd.Series, sample_size: int, random_state: int) -> np.ndarray:
    """Return deterministic proportional sample positions for a label series."""
    n_rows = len(strata)
    sample_size = min(max(1, int(sample_size)), n_rows)
    values = strata.reset_index(drop=True)
    groups = {key: np.asarray(pos, dtype=int) for key, pos in values.groupby(values).indices.items()}
    expected = {key: sample_size * len(indices) / n_rows for key, indices in groups.items()}
    allocation = {key: int(np.floor(value)) for key, value in expected.items()}
    remaining = sample_size - sum(allocation.values())
    for key in sorted(expected, key=lambda item: expected[item] - allocation[item], reverse=True):
        if remaining <= 0:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            remaining -= 1

    rng = np.random.default_rng(random_state)
    selected = [
        rng.choice(groups[key], size=allocation[key], replace=False)
        for key in groups
        if allocation[key] > 0
    ]
    indices = np.concatenate(selected) if selected else np.arange(n_rows)
    return np.sort(indices.astype(int))


def _cluster_metrics(x: np.ndarray, labels: np.ndarray, silhouette_size: int, random_state: int):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return None, None, None
    try:
        dbi = float(davies_bouldin_score(x, labels))
        chi = float(calinski_harabasz_score(x, labels))
        if len(labels) > silhouette_size:
            selected = _stratified_indices(pd.Series(labels), silhouette_size, random_state)
            silhouette = float(silhouette_score(x[selected], labels[selected]))
        else:
            silhouette = float(silhouette_score(x, labels))
        return round(silhouette, 4), round(dbi, 4), round(chi, 2)
    except ValueError:
        return None, None, None


def run_benchmark_matrix(
    x_base: np.ndarray,
    x_proposed: np.ndarray,
    features_df: pd.DataFrame,
    k: int = 4,
    eps: float = 1.4,
    min_samples: int = 18,
    dbscan_sample_size: int = 50_000,
    silhouette_sample_size: int = 5_000,
    random_state: int = 42,
):
    """Fit C0/C1 on all valid customers and C2 on a fixed stratified sample.

    Return C2 labels and source row positions only for the DBSCAN subsample, so
    dashboard consumers cannot mistake unsampled customers for noise.
    """
    n_rows = len(features_df)
    if n_rows != len(x_base) or n_rows != len(x_proposed):
        raise ValueError("Số hàng trong đặc trưng và ma trận mô hình không khớp.")
    if n_rows <= k:
        raise ValueError(f"k={k} cần nhỏ hơn số khách hàng hợp lệ ({n_rows}).")

    results = []
    labels = {}
    score_indices = _stratified_indices(
        features_df["Purchase_Flag"], min(silhouette_sample_size, n_rows), random_state
    )

    for code, matrix, config, feature_label in [
        ("C0", x_base, "C0 (Baseline RFM)", "RFM hiệu chỉnh (3 biến)"),
        ("C1", x_proposed, "C1 (Proposed + K-Means)", "RFM + hành vi (9 biến)"),
    ]:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=random_state,
        )
        cluster_labels = model.fit_predict(matrix)
        labels[code] = cluster_labels
        silhouette, dbi, chi = _cluster_metrics(
            matrix[score_indices], cluster_labels[score_indices],
            silhouette_sample_size, random_state,
        )
        results.append({
            "Cấu hình": config,
            "Thuật toán": "K-Means",
            "Đặc trưng": feature_label,
            "Silhouette (mẫu)": silhouette,
            "Davies–Bouldin": dbi,
            "Calinski–Harabasz": chi,
            "Tỷ lệ nhiễu": "0%",
            "Số cụm": int(len(np.unique(cluster_labels))),
            "Số khách chạy": n_rows,
        })

    dbscan_indices = _stratified_indices(
        features_df["Purchase_Flag"], min(dbscan_sample_size, n_rows), random_state
    )
    x_dbscan = x_proposed[dbscan_indices]
    dbscan_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(x_dbscan)
    labels["C2_sample"] = dbscan_labels
    labels["C2_indices"] = dbscan_indices

    is_noise = dbscan_labels == -1
    valid = ~is_noise
    cluster_count = len(np.unique(dbscan_labels[valid]))
    if cluster_count >= 2 and int(valid.sum()) >= 3:
        sc, dbi, chi = _cluster_metrics(
            x_dbscan[valid], dbscan_labels[valid], silhouette_sample_size, random_state
        )
    else:
        sc, dbi, chi = None, None, None
    noise_pct = float(is_noise.mean() * 100) if len(dbscan_labels) else 0.0
    results.append({
        "Cấu hình": "C2 (Proposed + DBSCAN)",
        "Thuật toán": "DBSCAN",
        "Đặc trưng": "RFM + hành vi (9 biến)",
        "Silhouette (mẫu)": sc,
        "Davies–Bouldin": dbi,
        "Calinski–Harabasz": chi,
        "Tỷ lệ nhiễu": f"{noise_pct:.2f}%",
        "Số cụm": int(cluster_count),
        "Số khách chạy": int(len(dbscan_indices)),
    })
    return pd.DataFrame(results), labels
