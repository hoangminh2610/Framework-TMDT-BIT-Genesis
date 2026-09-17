import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import pipeline as pl

st.set_page_config(
    page_title="Framework Phân Tích Hành Vi Khách Hàng Trên Nền Tảng Thương Mại Điện Tử",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] button {
        height: auto !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    [data-testid="stSidebar"] button p {
        white-space: pre-line !important;
        text-align: center !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Framework Phân Tích Hành Vi Khách Hàng Trên Nền Tảng TMĐT")
st.caption("Đề tài nghiên cứu khoa học - BIT Genesis 2026 | GVHD: TS. Võ Văn Hải")
st.markdown("Framework 6 tầng: **Big Data - Data Mining - Machine Learning**")

# Sidebar
st.sidebar.header("1. Cấu hình Dữ liệu")
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:",
    ("Dữ liệu mẫu giả lập REES46 (50.000 events)", "Tải file CSV từ máy")
)

uploaded_file = None
if data_source == "Tải file CSV từ máy":
    uploaded_file = st.sidebar.file_uploader("Tải file CSV từ Kaggle", type=['csv'])

st.sidebar.header("2. Siêu tham số Thực nghiệm")
k_clusters = st.sidebar.slider("Số cụm K-Means (k)", 2, 8, 4)
dbscan_eps = st.sidebar.slider("Bán kính ε (DBSCAN)", 0.5, 3.0, 1.4, step=0.1)
dbscan_min_samples = st.sidebar.slider("MinPts (DBSCAN)", 4, 25, 18)

run_button = st.sidebar.button("Khởi chạy Pipeline\nThực nghiệm", type="primary", use_container_width=True)

if run_button:
    is_mock = (data_source == "Dữ liệu mẫu giả lập REES46 (50.000 events)")
    
    if not is_mock and uploaded_file is None:
        st.error("Vui lòng tải file CSV lên.")
        st.stop()

    with st.spinner("Đang thực thi ETL & Feature Engineering..."):
        if is_mock:
            df_raw = pl.generate_mock_rees46_data()
        else:
            df_raw = pd.read_csv(uploaded_file)
            
        df_clean = pl.clean_raw_data(df_raw)
        features_df = pl.extract_features(df_clean)
        x_base, x_prop, base_cols, prop_cols = pl.prepare_datasets(features_df)

        benchmark_df, labels_dict = pl.run_benchmark_matrix(
            x_base, x_prop, 
            k=k_clusters, 
            eps=dbscan_eps, 
            min_samples=dbscan_min_samples
        )
        
        features_df['Cluster_C1'] = labels_dict['C1']
        features_df['Cluster_C2'] = labels_dict['C2']

    # Giao diện kết quả
    tab1, tab2, tab3, tab4 = st.tabs([
        "Tổng quan Dữ liệu", 
        "Đặc trưng Hành vi", 
        "Báo cáo Benchmark", 
        "Chân dung Khách hàng"
    ])

    with tab1:
        st.subheader("Thống kê phễu lọc và tiền xử lý dữ liệu (Data Funnel)")

        # 3 chỉ số tổng quan
        c1, c2, c3 = st.columns(3)
        c1.metric("Sự kiện thô", f"{len(df_raw):,}")
        c2.metric("Sự kiện sau lọc", f"{len(df_clean):,}")
        c3.metric("Khách hàng hợp lệ", f"{len(features_df):,}")

        st.divider()

        # Biểu đồ phân bố hành vi
        event_dist = df_clean["event_type"].value_counts().reset_index()
        event_dist.columns = ["Loại hành vi", "Số lượng"]

        fig_bar = px.bar(
            event_dist,
            x="Loại hành vi",
            y="Số lượng",
            color="Loại hành vi",
            title="Phễu tương tác",
        )

        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Bảng Data Funnel đưa xuống cuối
        st.subheader("Chi tiết các giai đoạn lọc dữ liệu")

        funnel_table = pl.get_data_funnel_stats(df_raw)

        st.dataframe(
            funnel_table,
            use_container_width=True,
            hide_index=True
    )

    with tab2:
        st.subheader("Bảng đặc trưng (10 dòng đầu)")
        st.dataframe(features_df.head(10), use_container_width=True)
        
        fig_rfm = px.scatter(
            features_df, x='Recency', y='Monetary', 
            color=features_df['Purchase_Flag'].astype(str),
            title="Tương quan Recency vs Monetary (Cờ Purchase)"
        )
        st.plotly_chart(fig_rfm, use_container_width=True)

    with tab3:
        st.subheader("Ma trận Benchmark Đối chuẩn 3 Cấu hình")
        st.dataframe(benchmark_df, use_container_width=True)
        
        pca = PCA(n_components=2, random_state=42)
        x_pca = pca.fit_transform(x_prop)
        features_df['PCA1'] = x_pca[:, 0]
        features_df['PCA2'] = x_pca[:, 1]
        
        col_pca1, col_pca2 = st.columns(2)
        with col_pca1:
            fig_p1 = px.scatter(features_df, x='PCA1', y='PCA2', color=features_df['Cluster_C1'].astype(str), title=f"K-Means (C1, k={k_clusters})")
            st.plotly_chart(fig_p1, use_container_width=True)
        with col_pca2:
            fig_p2 = px.scatter(features_df, x='PCA1', y='PCA2', color=features_df['Cluster_C2'].astype(str), title="DBSCAN (C2) - Nhãn -1 là Nhiễu")
            st.plotly_chart(fig_p2, use_container_width=True)

    with tab4:
        st.subheader("Chân dung từng Cụm (Theo K-Means C1)")
        cluster_profile = features_df.groupby('Cluster_C1')[prop_cols].mean().reset_index()
        st.dataframe(cluster_profile.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

else:
    st.info("Bấm nút 'Khởi chạy Pipeline Thực nghiệm' ở cột bên trái để bắt đầu phân tích mô hình.")