import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def generate_mock_rees46_data(n_events=50000, n_users=2000):
  """Tự động sinh dữ liệu giả lập chuẩn cấu trúc dataset REES46 Kaggle để chạy thử nghiệm."""
  np.random.seed(42)
  users = [f'user_{i}' for i in range(1, n_users + 1)]
  event_types = ['view', 'cart', 'remove_from_cart', 'purchase']
  p_types = [0.70, 0.18, 0.08, 0.04]

  timestamps = pd.date_range(
      start='2019-10-01', end='2020-02-28', periods=n_events
  )

  df = pd.DataFrame({
      'event_time': timestamps,
      'event_type': np.random.choice(event_types, size=n_events, p=p_types),
      'product_id': np.random.randint(1000, 5000, size=n_events),
      'category_id': np.random.randint(10, 50, size=n_events),
      'brand': np.random.choice(
          ['loreal', 'maybelline', 'innisfree', 'mac', 'unknown'], size=n_events
      ),
      'price': np.round(np.random.exponential(scale=25, size=n_events) + 1.5, 2),
      'user_id': np.random.choice(users, size=n_events),
      'user_session': [f'sess_{np.random.randint(1, 10000)}' for _ in range(n_events)],
  })
  return df


def clean_raw_data(df):
  """Bước 4.4: Lọc dữ liệu thô và loại bỏ người dùng không đủ 2 sự kiện."""
  df['event_time'] = pd.to_datetime(df['event_time'])
  df = df[df['price'] > 0]
  df = df.dropna(subset=['user_id'])
  df = df.drop_duplicates(
      subset=['event_time', 'user_id', 'product_id', 'event_type']
  )

  # Lọc người dùng có tối thiểu 2 tương tác
  user_counts = df['user_id'].value_counts()
  valid_users = user_counts[user_counts >= 2].index
  df_clean = df[df['user_id'].isin(valid_users)].copy()
  return df_clean


def extract_features(df_clean):
  """Bước 4.5: Trích xuất tập Baseline RFM hiệu chỉnh và Proposed Features."""
  t_ref = df_clean['event_time'].max()
  penalty_days = 150.0  # Chu kỳ 5 tháng làm mốc phạt cho người chưa mua

  user_groups = df_clean.groupby('user_id')

  records = []
  for user_id, group in user_groups:
    purchases = group[group['event_type'] == 'purchase']
    f = len(purchases)
    m = purchases['price'].sum() if f > 0 else 0.0

    if f > 0:
      r = (t_ref - purchases['event_time'].max()).total_seconds() / 86400.0
      p_flag = 1
    else:
      r = (
          (t_ref - group['event_time'].max()).total_seconds() / 86400.0
      ) + penalty_days
      p_flag = 0

    view_cnt = (group['event_type'] == 'view').sum()
    cart_cnt = (group['event_type'] == 'cart').sum()

    if cart_cnt > 0:
      cart_to_purchase_gap = 1.0 - min(1.0, f / cart_cnt)
    else:
      cart_to_purchase_gap = 0.0

    span = (
        group['event_time'].max() - group['event_time'].min()
    ).total_seconds() / 86400.0
    total_sess = group['user_session'].nunique()

    records.append({
        'user_id': user_id,
        'Recency': r,
        'Frequency': f,
        'Monetary': m,
        'Purchase_Flag': p_flag,
        'View_Count': view_cnt,
        'Cart_Count': cart_cnt,
        'Cart_to_Purchase_Gap': cart_to_purchase_gap,
        'Interaction_Span': span,
        'Total_Sessions': total_sess,
    })

  features_df = pd.DataFrame(records)
  return features_df


def prepare_datasets(features_df):
  """Bước 4.5.C: Áp dụng Log(x + 1) và chuẩn hóa qua StandardScaler."""
  baseline_cols = ['Recency', 'Frequency', 'Monetary']
  proposed_cols = [
      'Recency',
      'Frequency',
      'Monetary',
      'Purchase_Flag',
      'View_Count',
      'Cart_Count',
      'Cart_to_Purchase_Gap',
      'Interaction_Span',
      'Total_Sessions',
  ]

  # Log transformation cho các biến lệch
  df_log = features_df.copy()
  log_candidate = [
      'Recency',
      'Frequency',
      'Monetary',
      'View_Count',
      'Cart_Count',
      'Interaction_Span',
      'Total_Sessions',
  ]
  for c in log_candidate:
    df_log[c] = np.log1p(df_log[c])

  scaler_base = StandardScaler()
  x_base = scaler_base.fit_transform(df_log[baseline_cols])

  scaler_prop = StandardScaler()
  x_prop = scaler_prop.fit_transform(df_log[proposed_cols])

  return x_base, x_prop, baseline_cols, proposed_cols


def run_benchmark_matrix(x_base, x_prop, k=4, eps=1.2, min_samples=10):
  """Thực thi ma trận thực nghiệm 3 cấu hình C0, C1, C2."""
  results = []
  labels_dict = {}

  # Cấu hình 0: Baseline RFM + K-Means
  km_c0 = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300, random_state=42)
  labels_c0 = km_c0.fit_predict(x_base)
  labels_dict['C0'] = labels_c0
  results.append({
      'Cấu hình': 'C0 (Baseline RFM)',
      'Thuật toán': 'K-Means',
      'Đặc trưng': 'RFM Hiệu chỉnh (3 biến)',
      'Silhouette (SC)': round(silhouette_score(x_base, labels_c0), 4),
      'Davies-Bouldin (DBI)': round(davies_bouldin_score(x_base, labels_c0), 4),
      'Calinski-Harabasz (CHI)': round(calinski_harabasz_score(x_base, labels_c0), 2),
      'Tỷ lệ nhiễu (Noise %)': '0.0%',
      'Số cụm': k,
  })

  # Cấu hình 1: Proposed Features + K-Means
  km_c1 = KMeans(n_clusters=k, init='k-means++', n_init=20, max_iter=300, random_state=42)
  labels_c1 = km_c1.fit_predict(x_prop)
  labels_dict['C1'] = labels_c1
  results.append({
      'Cấu hình': 'C1 (Proposed + K-Means)',
      'Thuật toán': 'K-Means',
      'Đặc trưng': 'RFM + Hành vi (9 biến)',
      'Silhouette (SC)': round(silhouette_score(x_prop, labels_c1), 4),
      'Davies-Bouldin (DBI)': round(davies_bouldin_score(x_prop, labels_c1), 4),
      'Calinski-Harabasz (CHI)': round(calinski_harabasz_score(x_prop, labels_c1), 2),
      'Tỷ lệ nhiễu (Noise %)': '0.0%',
      'Số cụm': k,
  })

  # Cấu hình 2: Proposed Features + DBSCAN
  db = DBSCAN(eps=eps, min_samples=min_samples)
  labels_c2 = db.fit_predict(x_prop)
  labels_dict['C2'] = labels_c2

  n_noise = (labels_c2 == -1).sum()
  noise_pct = (n_noise / len(labels_c2)) * 100.0
  valid_mask = labels_c2 != -1
  unique_clusters = set(labels_c2[valid_mask])

  if len(unique_clusters) >= 2:
    sc_c2 = round(silhouette_score(x_prop[valid_mask], labels_c2[valid_mask]), 4)
    dbi_c2 = round(davies_bouldin_score(x_prop[valid_mask], labels_c2[valid_mask]), 4)
    chi_c2 = round(calinski_harabasz_score(x_prop[valid_mask], labels_c2[valid_mask]), 2)
  else:
    sc_c2, dbi_c2, chi_c2 = None, None, None

  results.append({
      'Cấu hình': 'C2 (Proposed + DBSCAN)',
      'Thuật toán': 'DBSCAN',
      'Đặc trưng': 'RFM + Hành vi (9 biến)',
      'Silhouette (SC)': sc_c2,
      'Davies-Bouldin (DBI)': dbi_c2,
      'Calinski-Harabasz (CHI)': chi_c2,
      'Tỷ lệ nhiễu (Noise %)': f'{noise_pct:.2f}%',
      'Số cụm': len(unique_clusters),
  })

  return pd.DataFrame(results), labels_dict