from __future__ import annotations

from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

import pipeline as pl
import ui


st.set_page_config(
    page_title="Customer Behavior Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
)

ui.inject_styles()
st.markdown(
    """
    <style>
    /* Keep Streamlit's sidebar toggle ligature on its icon font after app reruns. */
    .stApp [data-testid="stIconMaterial"] {
      font-family: "Material Symbols Rounded" !important;
      font-feature-settings: "liga";
      font-style: normal;
      font-weight: normal;
      letter-spacing: normal;
      line-height: 1;
      text-transform: none;
      white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _segment_profile(features: pd.DataFrame) -> pd.DataFrame:
    """Create descriptive, explicitly heuristic labels for K-Means clusters."""
    profile = features.groupby("Cluster_C1", as_index=False).agg(
        So_khach_hang=("user_id", "size"),
        Ty_le=("user_id", "size"),
        Recency_tb=("Recency", "mean"),
        Frequency_tb=("Frequency", "mean"),
        Monetary_tb=("Monetary", "mean"),
        Ti_le_da_mua=("Purchase_Flag", "mean"),
        View_tb=("View_Count", "mean"),
        Cart_tb=("Cart_Count", "mean"),
        Gap_tb=("Cart_to_Purchase_Gap", "mean"),
        Span_tb=("Interaction_Span", "mean"),
        Session_tb=("Total_Sessions", "mean"),
    )
    profile["Ty_le"] = profile["So_khach_hang"] / max(1, len(features))

    med_frequency = features["Frequency"].median()
    high_frequency = features["Frequency"].quantile(0.75)
    high_monetary = features["Monetary"].quantile(0.75)
    dormant_recency = features["Recency"].quantile(0.75)
    median_views = features["View_Count"].median()

    labels, actions = [], []
    for row in profile.itertuples(index=False):
        if row.Ti_le_da_mua < 0.2:
            if row.Cart_tb >= 1:
                labels.append("Đã quan tâm, chưa mua")
                actions.append("Thử nhắc giỏ hàng; cân nhắc nội dung hỗ trợ quyết định mua.")
            elif row.View_tb >= median_views:
                labels.append("Đang khám phá, chưa mua")
                actions.append("Gợi ý nội dung và sản phẩm liên quan; đo lường chuyển đổi trước khi khuyến mãi.")
            else:
                labels.append("Tương tác thấp, chưa mua")
                actions.append("Theo dõi thêm dữ liệu trước khi phân bổ ngân sách tiếp thị.")
        elif row.Recency_tb >= dormant_recency:
            labels.append("Người mua có dấu hiệu ngủ đông")
            actions.append("Đây là cờ theo Recency, không phải dự báo churn; thử chiến dịch tái tương tác có kiểm soát.")
        elif row.Frequency_tb >= high_frequency and row.Monetary_tb >= high_monetary:
            labels.append("Nhóm mua lặp lại, giá trị cao")
            actions.append("Ưu tiên giữ chân và giới thiệu sản phẩm bổ trợ; chưa suy diễn CLV.")
        elif row.Frequency_tb > med_frequency:
            labels.append("Khách mua lặp lại")
            actions.append("Thử cross-sell phù hợp và theo dõi tỷ lệ mua lại.")
        else:
            labels.append("Khách đã mua, tần suất thấp")
            actions.append("Khuyến khích lần mua tiếp theo và theo dõi Recency.")

    profile["Phân khúc gợi ý"] = labels
    profile["Hướng hành động tham khảo"] = actions
    return profile


@st.cache_data(ttl=900, max_entries=1, show_spinner=False)
def analyze_data(
    source_name: str,
    csv_bytes: bytes | None,
    processing_engine: str,
    k: int,
    eps: float,
    min_samples: int,
    dbscan_sample_size: int,
):
    """Run an analysis only after form submission; retain its result in cache."""
    if source_name == "Dữ liệu minh họa tổng hợp":
        raw = pl.generate_mock_rees46_data()
        funnel = pl.get_data_funnel_stats(raw)
        cleaned = pl.clean_raw_data(raw)
        if cleaned.empty:
            raise ValueError("Không còn sự kiện hợp lệ sau bước làm sạch.")
        event_counts = cleaned["event_type"].value_counts().rename_axis("Loại sự kiện").reset_index(name="Số sự kiện")
        features = pl.extract_features(cleaned)
        raw_event_count = int(len(raw))
        clean_event_count = int(len(cleaned))
        period_start, period_end = cleaned["event_time"].min(), cleaned["event_time"].max()
    else:
        if not csv_bytes:
            raise ValueError("Hãy tải lên tệp CSV trước khi chạy phân tích.")
        if processing_engine == "Spark local[*]":
            spark_result = pl.process_csv_with_spark(csv_bytes)
            funnel = spark_result["funnel"]
            event_counts = spark_result["event_counts"]
            features = spark_result["features"]
            raw_event_count = spark_result["raw_event_count"]
            clean_event_count = spark_result["clean_event_count"]
            period_start, period_end = spark_result["period_start"], spark_result["period_end"]
        else:
            try:
                raw = pd.read_csv(BytesIO(csv_bytes), low_memory=False)
            except Exception as exc:
                raise ValueError(f"Không đọc được tệp CSV: {exc}") from exc
            pl.validate_raw_data(raw)
            funnel = pl.get_data_funnel_stats(raw)
            cleaned = pl.clean_raw_data(raw)
            if cleaned.empty:
                raise ValueError("Không còn sự kiện hợp lệ sau bước làm sạch.")
            event_counts = cleaned["event_type"].value_counts().rename_axis("Loại sự kiện").reset_index(name="Số sự kiện")
            features = pl.extract_features(cleaned)
            raw_event_count = int(len(raw))
            clean_event_count = int(len(cleaned))
            period_start, period_end = cleaned["event_time"].min(), cleaned["event_time"].max()

    x_base, x_proposed, _, proposed_cols = pl.prepare_datasets(features)
    benchmark, labels = pl.run_benchmark_matrix(
        x_base,
        x_proposed,
        features,
        k=k,
        eps=eps,
        min_samples=min_samples,
        dbscan_sample_size=dbscan_sample_size,
    )
    features["Cluster_C0"] = labels["C0"]
    features["Cluster_C1"] = labels["C1"]
    features["Cluster_C2"] = -2  # -2 means this customer was not in the DBSCAN sample.
    dbscan_indices = labels["C2_indices"]
    features.loc[dbscan_indices, "Cluster_C2"] = labels["C2_sample"]

    profile = _segment_profile(features)
    name_by_cluster = dict(zip(profile["Cluster_C1"], profile["Phân khúc gợi ý"]))
    features["Phân khúc gợi ý"] = features["Cluster_C1"].map(name_by_cluster)

    # Keep charts legible and ensure C1 and DBSCAN views show the same sampled users.
    rng = np.random.default_rng(42)
    display_count = min(3_000, len(dbscan_indices))
    display_indices = np.sort(rng.choice(dbscan_indices, size=display_count, replace=False))
    projection = PCA(n_components=2, random_state=42).fit_transform(x_proposed[display_indices])
    visual = features.iloc[display_indices][
        ["user_id", "Recency", "Frequency", "Monetary", "Cart_Count", "Cluster_C1", "Cluster_C2", "Phân khúc gợi ý"]
    ].copy()
    visual["PCA 1"], visual["PCA 2"] = projection[:, 0], projection[:, 1]
    visual["Nhãn DBSCAN"] = visual["Cluster_C2"].map(
        lambda value: "Chưa phân cụm" if value == -2 else "Nhiễu" if value == -1 else f"Cụm {value}"
    )

    known_events = {"view", "cart", "remove_from_cart", "purchase"}
    unknown_event_types = sorted(set(event_counts["Loại sự kiện"].dropna()) - known_events)
    metadata = {
        "source": source_name,
        "engine": processing_engine if source_name != "Dữ liệu minh họa tổng hợp" else "Pandas demo",
        "raw_event_count": raw_event_count,
        "clean_event_count": clean_event_count,
        "customer_count": int(len(features)),
        "period_start": period_start,
        "period_end": period_end,
        "dbscan_sample_count": int(len(dbscan_indices)),
        "run_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "unknown_event_types": unknown_event_types,
        "proposed_columns": proposed_cols,
    }
    return {
        "features": features,
        "benchmark": benchmark,
        "profile": profile,
        "funnel": funnel,
        "event_counts": event_counts,
        "visual": visual,
        "metadata": metadata,
    }


def _format_money(value: float) -> str:
    return f"${value:,.2f}"


def _safe_plot(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=8, r=8, t=56, b=12),
        legend_title_text="",
        font=dict(family="Inter, ui-sans-serif, sans-serif", color="#17243a"),
        colorway=["#4355d8", "#16845b", "#d18a25", "#61738d", "#a05d99", "#2c8b9b"],
        paper_bgcolor="white", plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(color="#17243a")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f5", zeroline=False, linecolor="#e3e8f0")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="#e3e8f0")
    return fig


# Product navigation and analysis controls
_navigation_migration = {
    "Tổng quan": "Overview",
    "Pipeline dữ liệu": "Data Pipeline",
    "Phân khúc khách": "Segmentation",
    "So sánh mô hình": "Models",
}
if st.session_state.get("page_navigation") in _navigation_migration:
    st.session_state["page_navigation"] = _navigation_migration[st.session_state["page_navigation"]]

with st.sidebar:
    ui.render_brand()
    st.markdown("**Workspace**")
    view_name = st.radio(
        "Điều hướng",
        ["Overview", "Data Pipeline", "Segmentation", "Models", "Insights"],
        key="page_navigation",
        label_visibility="collapsed",
    )
    st.markdown("**Thiết lập phân tích**")
    source_name = st.selectbox(
        "Nguồn dữ liệu",
        ["Dữ liệu minh họa tổng hợp", "Tải CSV REES46 từ máy"],
        help="Dữ liệu minh họa được sinh tổng hợp, không phải bản ghi thực từ Kaggle.",
    )
    processing_engine = st.selectbox(
        "Công cụ ETL",
        ["Pandas (tệp nhỏ)", "Spark local[*]"],
        disabled=source_name == "Dữ liệu minh họa tổng hợp",
        help="Spark chạy trên máy hiện tại; cần PySpark và Java tương thích. Đây không phải cụm Spark phân tán.",
    )
    uploaded_file = None
    if source_name == "Tải CSV REES46 từ máy":
        uploaded_file = st.file_uploader(
            "Tệp CSV",
            type=["csv"],
            help="Cần event_time, event_type, product_id, price, user_id và user_session.",
        )
    dbscan_sample_cap = 1_700 if source_name == "Dữ liệu minh họa tổng hợp" else 50_000
    dbscan_sample_min = min(1_000, dbscan_sample_cap)
    with st.form("analysis_form", border=True):
        st.markdown("**Tham số mô hình**")
        k_clusters = st.slider("Số cụm K-Means (k)", 2, 8, 4)
        dbscan_eps = st.slider("Bán kính DBSCAN (ε)", 0.5, 3.0, 1.4, step=0.1)
        dbscan_min_samples = st.slider("DBSCAN MinPts", 4, 25, 18)
        dbscan_sample_size = st.slider(
            "Mẫu DBSCAN (khách hợp lệ)",
            min_value=dbscan_sample_min,
            max_value=dbscan_sample_cap,
            value=dbscan_sample_cap,
            step=100 if dbscan_sample_cap == 1_700 else 1_000,
            key="dbscan_sample_size_demo" if dbscan_sample_cap == 1_700 else "dbscan_sample_size_csv",
            help=(
                "Demo có 50.000 sự kiện và 2.000 người dùng ban đầu; sau lọc còn 1.700 khách hợp lệ, "
                "nên DBSCAN tối đa dùng 1.700 khách."
                if dbscan_sample_cap == 1_700
                else "Lấy tối đa 50.000 khách hàng hợp lệ theo mẫu phân tầng; đây không phải số sự kiện."
            ),
        )
        submitted = st.form_submit_button(
            ":material/play_arrow: Chạy pipeline", type="primary", width="stretch"
        )
    st.caption(
        "Demo: 50.000 sự kiện → 2.000 người dùng ban đầu → tối đa 1.700 khách hợp lệ sau lọc."
        if source_name == "Dữ liệu minh họa tổng hợp"
        else "Cấu hình theo luận văn: k = 4; DBSCAN lấy mẫu tối đa 50.000 khách hàng hợp lệ."
    )
    st.markdown("---")
    st.caption("CUSTOMER BEHAVIOR LAB · BIT GENESIS RESEARCH AWARDS 2026")

if submitted:
    csv_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    st.session_state.pop("analysis_results", None)
    try:
        with st.spinner("Đang làm sạch, tạo đặc trưng và chạy ba cấu hình…"):
            st.session_state["analysis_results"] = analyze_data(
                source_name,
                csv_bytes,
                processing_engine,
                k_clusters,
                dbscan_eps,
                dbscan_min_samples,
                dbscan_sample_size,
            )
        st.session_state["analysis_settings"] = {
            "source": source_name,
            "engine": processing_engine,
            "k": k_clusters,
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "dbscan_sample_size": dbscan_sample_size,
        }
        st.toast("Phân tích hoàn tất.", icon=":material/check_circle:")
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Pipeline gặp lỗi ({type(exc).__name__}): {exc}")

analysis = st.session_state.get("analysis_results")
page_copy = {
    "Overview": ("Overview", "Hệ thống phân tích dữ liệu và hành vi khách hàng trên nền tảng thương mại điện tử"),
    "Data Pipeline": ("Data pipeline", "Theo dõi tác động của từng bước làm sạch lên số sự kiện và số khách hàng."),
    "Segmentation": ("Customer segmentation", "Đọc quy mô, hành vi và đặc trưng của các cụm C1."),
    "Models": ("Model comparison", "Đối chiếu ba cấu hình nghiên cứu và xem từng chỉ số đánh giá."),
    "Insights": ("Research insights", "Các gợi ý diễn giải và danh sách khách hàng có thể lọc, xuất."),
}
category, description = page_copy[view_name]
if analysis is None:
    ui.render_page_header(category, "Customer Behavior Lab", description)
    with st.container(border=True):
        st.markdown("### Thiết lập phân tích")
        st.write("Lựa chọn tập dữ liệu và các tham số phân tích để khởi chạy quy trình xử lý và phân tích")
        st.caption("Tập dữ liệu mô phỏng gồm 50.000 sự kiện và 2.000 người dùng, được xây dựng dựa trên cấu trúc dữ liệu hành vi thương mại điện tử và sử dụng cho mục đích thực nghiệm.")
    st.stop()

features = analysis["features"]
benchmark = analysis["benchmark"]
profile = analysis["profile"]
metadata = analysis["metadata"]
event_counts = analysis["event_counts"]
source_label = "Dữ liệu minh họa" if "minh họa" in metadata["source"].lower() else "CSV đã tải"
ui.render_page_header(
    category,
    view_name,
    description,
    [
        source_label,
        f"{metadata['customer_count']:,} khách hợp lệ",
        f"{metadata['clean_event_count']:,} sự kiện sau làm sạch",
        f"Cập nhật {metadata['run_at']}",
    ],
)
if metadata["unknown_event_types"]:
    st.warning("Có loại sự kiện ngoài view/cart/remove_from_cart/purchase: " + ", ".join(metadata["unknown_event_types"]))

if view_name == "Overview":
    buyer_rate = float(features["Purchase_Flag"].mean())
    mean_revenue = float(features["Monetary"].mean())
    metrics = st.columns(4)
    with metrics[0]:
        ui.render_metric_card("Khách hàng được phân tích", f"{len(features):,}", "Khách còn lại sau bước làm sạch", "◉")
    with metrics[1]:
        ui.render_metric_card("Sự kiện sau làm sạch", f"{metadata['clean_event_count']:,}", "Bản ghi hành vi còn hợp lệ", "↗")
    with metrics[2]:
        ui.render_metric_card("Khách từng mua", f"{buyer_rate:.1%}", "Tính trên khách hàng được phân tích", "✓")
    with metrics[3]:
        ui.render_metric_card("Chi tiêu TB / khách", _format_money(mean_revenue), "Monetary trung bình", "$ ")

    ui.render_section_header("Hành vi và phân khúc", "Tổng hợp sự kiện quan sát được và quy mô từng cụm K-Means C1.")
    left, right = st.columns([1, 1.15])
    with left:
        event_chart_data = event_counts.copy()
        event_chart_data["Tỷ trọng"] = event_chart_data["Số sự kiện"] / max(1, event_chart_data["Số sự kiện"].sum())
        event_chart_data = event_chart_data.sort_values("Số sự kiện")
        event_colors = {"view": "#4355d8", "cart": "#d18a25", "remove_from_cart": "#9aa6b7", "purchase": "#16845b"}
        event_fig = px.bar(
            event_chart_data, x="Số sự kiện", y="Loại sự kiện", orientation="h",
            color="Loại sự kiện", text=event_chart_data["Tỷ trọng"].map(lambda value: f"{value:.1%}"),
            color_discrete_map=event_colors,
        )
        event_fig.update_layout(showlegend=False, xaxis_title="Số sự kiện", yaxis_title="", height=300)
        ui.render_chart_card("Cơ cấu sự kiện", "Tỷ trọng theo loại hành vi; không phải tỷ lệ chuyển đổi.", _safe_plot(event_fig))
    with right:
        size_fig = px.bar(
            profile.sort_values("So_khach_hang"),
            x="So_khach_hang", y="Phân khúc gợi ý", orientation="h",
            color=profile.sort_values("So_khach_hang")["Cluster_C1"].astype(str), text="So_khach_hang",
            color_discrete_sequence=["#4355d8", "#16845b", "#d18a25", "#61738d", "#a05d99", "#2c8b9b"],
        )
        size_fig.update_traces(textposition="outside", cliponaxis=False)
        size_fig.update_layout(showlegend=False, xaxis_title="Số khách hàng", yaxis_title="", height=320)
        ui.render_chart_card("Quy mô phân khúc", "Số khách và tỷ trọng theo nhãn diễn giải C1.", _safe_plot(size_fig))

    largest = profile.sort_values("So_khach_hang", ascending=False).iloc[0]
    st.markdown(
        f'<div class="cbl-callout"><strong>Điểm cần chú ý</strong> · Phân khúc lớn nhất là {largest["Phân khúc gợi ý"]}, chiếm {largest["Ty_le"]:.1%} số khách được phân tích. Nhãn phân khúc là gợi ý diễn giải dựa trên trung bình cụm.</div>',
        unsafe_allow_html=True,
    )
    ui.render_section_header("Không gian hành vi RFM", "Mỗi điểm là một khách hàng trong mẫu trực quan hóa tối đa 3.000 khách; màu biểu thị nhãn gợi ý của C1.")
    visual = analysis["visual"]
    rfm_fig = px.scatter_3d(
        visual, x="Recency", y="Frequency", z="Monetary", color="Phân khúc gợi ý",
        hover_data=["user_id", "Cluster_C1", "Cart_Count"], opacity=0.76,
        color_discrete_sequence=["#4355d8", "#16845b", "#d18a25", "#61738d", "#a05d99", "#2c8b9b"],
    )
    rfm_fig.update_layout(height=500, scene=dict(
        xaxis_title="Recency · ngày", yaxis_title="Frequency · số lần mua", zaxis_title="Monetary · USD"
    ))
    ui.render_chart_card("Hành vi khách hàng trong không gian RFM", "Phép chiếu giữ nguyên giá trị Recency, Frequency và Monetary của mẫu hiển thị.", _safe_plot(rfm_fig), height=500)

elif view_name == "Data Pipeline":
    ui.render_section_header("Dòng xử lý dữ liệu", "Mỗi giai đoạn hiển thị riêng số sự kiện, số khách hàng và tỷ lệ khách được giữ lại.")
    funnel = analysis["funnel"].reset_index(drop=True)
    stage_columns = st.columns(len(funnel))
    stage_titles = ["Dữ liệu đầu vào", "Loại sự kiện thiếu / lỗi", "Khử trùng lặp", "Giữ khách từ 2 sự kiện"]
    for idx, (_, row) in enumerate(funnel.iterrows()):
        with stage_columns[idx]:
            ui.render_pipeline_card(
                idx + 1,
                stage_titles[idx] if idx < len(stage_titles) else str(row["Giai đoạn"]),
                f"{int(row['Số sự kiện']):,}",
                f"{int(row['Số khách hàng']):,}",
                f"{row['Tỷ lệ khách giữ lại']:.2%}",
            )
    st.caption(f"Tập phân tích hiện có {metadata['customer_count']:,} khách hàng hợp lệ và {metadata['clean_event_count']:,} sự kiện sau làm sạch.")
    with st.expander("Xem bảng số liệu chi tiết"):
        st.dataframe(
            funnel,
            column_config={
                "Tỷ lệ khách giữ lại": st.column_config.NumberColumn(format="percent"),
                "Số sự kiện": st.column_config.NumberColumn(format="localized"),
                "Số khách hàng": st.column_config.NumberColumn(format="localized"),
            },
            hide_index=True,
            width="stretch",
        )
    if "minh họa" in metadata["source"].lower():
        st.info("Dữ liệu đầu vào là dữ liệu tổng hợp để trình diễn: 50.000 sự kiện, 2.000 người dùng ban đầu. Số khách chạy DBSCAN lấy từ tập khách hợp lệ sau lọc, không phải số sự kiện.", icon=":material/info:")

elif view_name == "Segmentation":
    buyers = int(features["Purchase_Flag"].sum())
    kpis = st.columns(3)
    with kpis[0]:
        ui.render_metric_card("Số phân khúc C1", f"{profile['Cluster_C1'].nunique():,}", "K-Means trên RFM + hành vi")
    with kpis[1]:
        ui.render_metric_card("Khách được phân nhóm", f"{len(features):,}", "C1 gán nhãn cho toàn bộ khách hợp lệ")
    with kpis[2]:
        ui.render_metric_card("Khách từng mua", f"{buyers:,}", f"{buyers / max(1, len(features)):.1%} trong tập phân tích")
    ui.render_section_header("Chân dung các cụm", "Các nhãn và hướng hành động là heuristic để hỗ trợ đọc kết quả, cần chuyên gia rà soát.")
    segment_rows = list(profile.sort_values("Cluster_C1").iterrows())
    for offset in range(0, len(segment_rows), 2):
        columns = st.columns(2)
        for col, (_, row) in zip(columns, segment_rows[offset:offset + 2]):
            with col:
                ui.render_segment_card(
                    int(row["Cluster_C1"]), str(row["Phân khúc gợi ý"]),
                    f"{int(row['So_khach_hang']):,}", f"{row['Ty_le']:.1%}",
                    f"{row['Ti_le_da_mua']:.1%}", _format_money(float(row["Monetary_tb"])),
                    f"{row['Recency_tb']:.1f}", str(row["Hướng hành động tham khảo"]),
                )
    visual = analysis["visual"]
    pca_fig = px.scatter(
        visual, x="PCA 1", y="PCA 2", color=visual["Cluster_C1"].astype(str),
        hover_data=["user_id", "Phân khúc gợi ý"], opacity=0.72,
        color_discrete_sequence=["#4355d8", "#16845b", "#d18a25", "#61738d", "#a05d99", "#2c8b9b"],
    )
    pca_fig.update_layout(xaxis_title="PCA 1", yaxis_title="PCA 2", height=440, legend_title_text="Cụm C1")
    ui.render_chart_card("Phân bố cụm trên PCA", "Mẫu điểm khớp mẫu dùng trong trực quan hóa; PCA chỉ là phép chiếu 2D để quan sát.", _safe_plot(pca_fig), height=440)
    radar_columns = ["Recency_tb", "Frequency_tb", "Monetary_tb", "View_tb", "Cart_tb", "Span_tb", "Session_tb"]
    radar_labels = ["Recency", "Frequency", "Monetary", "Views", "Cart", "Span", "Sessions"]
    denominator = (profile[radar_columns].max() - profile[radar_columns].min()).replace(0, 1)
    normalized = (profile[radar_columns] - profile[radar_columns].min()) / denominator
    radar_fig = go.Figure()
    for idx, row in profile.iterrows():
        values = normalized.loc[idx].tolist()
        radar_fig.add_trace(go.Scatterpolar(
            r=values + [values[0]], theta=radar_labels + [radar_labels[0]],
            name=f"Cụm {int(row['Cluster_C1'])} · {row['Phân khúc gợi ý']}", fill="toself", opacity=0.3,
        ))
    radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=480)
    ui.render_chart_card("So sánh đặc trưng tương đối", "Các biến được chuẩn hóa min–max để so sánh tương đối giữa cụm; không phải giá trị gốc.", _safe_plot(radar_fig), height=480)
    profile_view = profile.rename(columns={
        "Cluster_C1": "Cụm C1", "So_khach_hang": "Số khách hàng", "Ty_le": "Tỷ trọng khách",
        "Recency_tb": "Recency TB (ngày)", "Frequency_tb": "Frequency TB (lần mua)",
        "Monetary_tb": "Monetary TB (USD)", "Ti_le_da_mua": "Tỷ lệ từng mua",
        "View_tb": "Lượt xem TB", "Cart_tb": "Lượt thêm giỏ TB", "Gap_tb": "Cart-to-purchase gap TB",
        "Span_tb": "Chu kỳ tương tác TB (ngày)", "Session_tb": "Phiên TB",
    })
    with st.expander("Bảng đặc trưng đầy đủ"):
        st.dataframe(profile_view, column_config={
            "Tỷ trọng khách": st.column_config.NumberColumn(format="percent"),
            "Tỷ lệ từng mua": st.column_config.NumberColumn(format="percent"),
            "Monetary TB (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Recency TB (ngày)": st.column_config.NumberColumn(format="%.1f"),
            "Frequency TB (lần mua)": st.column_config.NumberColumn(format="%.2f"),
            "Hướng hành động tham khảo": st.column_config.TextColumn(width="large"),
        }, hide_index=True, width="stretch")

elif view_name == "Models":
    ui.render_section_header("Ba cấu hình nghiên cứu", "C0–C1 thay đổi bộ đặc trưng; C1–C2 dùng cùng bộ đặc trưng và đổi thuật toán.")
    sil = pd.to_numeric(benchmark["Silhouette (mẫu)"], errors="coerce")
    dbi = pd.to_numeric(benchmark["Davies–Bouldin"], errors="coerce")
    chi = pd.to_numeric(benchmark["Calinski–Harabasz"], errors="coerce")
    best_sil = benchmark.loc[sil.idxmax()] if sil.notna().any() else None
    best_dbi = benchmark.loc[dbi.idxmin()] if dbi.notna().any() else None
    best_chi = benchmark.loc[chi.idxmax()] if chi.notna().any() else None
    highlights = st.columns(3)
    for col, row, label, value, detail in [
        (highlights[0], best_sil, "Silhouette cao nhất", "Silhouette (mẫu)", "Giá trị cao hơn thường biểu thị cụm tách biệt hơn."),
        (highlights[1], best_dbi, "Davies–Bouldin thấp nhất", "Davies–Bouldin", "Giá trị thấp hơn thường tốt hơn."),
        (highlights[2], best_chi, "Calinski–Harabasz cao nhất", "Calinski–Harabasz", "Giá trị cao hơn thường tốt hơn."),
    ]:
        with col:
            ui.render_highlight(label, str(row["Cấu hình"]) if row is not None else "Chưa xác định", f"{row[value]:.4f}" if row is not None else "N/A", detail)
    st.caption("Các chỉ số được tính trên quần thể/mẫu đánh giá khác nhau: C0/C1 chấm trên tối đa 5.000 khách; C2 chấm sau khi bỏ nhiễu DBSCAN. Không diễn giải một chỉ số đơn lẻ là mô hình tốt nhất toàn diện.")
    with st.container(border=True):
        st.dataframe(
            benchmark,
            column_config={
                "Silhouette (mẫu)": st.column_config.NumberColumn(format="%.4f"),
                "Davies–Bouldin": st.column_config.NumberColumn(format="%.4f"),
                "Calinski–Harabasz": st.column_config.NumberColumn(format="%.2f"),
                "Số khách chạy": st.column_config.NumberColumn(format="localized"),
            },
            hide_index=True,
            width="stretch",
        )
    count_rows = []
    for config_name, cluster_column in [("C0 · RFM", "Cluster_C0"), ("C1 · RFM + hành vi", "Cluster_C1")]:
        counts = features[cluster_column].value_counts().sort_index()
        count_rows.extend({"Cấu hình": config_name, "Cụm": f"Cụm {int(cluster)}", "Số khách": int(count)} for cluster, count in counts.items())
    count_fig = px.bar(
        pd.DataFrame(count_rows), x="Cụm", y="Số khách", color="Cấu hình", barmode="group",
        color_discrete_sequence=["#4355d8", "#16845b"],
    )
    count_fig.update_layout(height=350, xaxis_title="", yaxis_title="Số khách hàng")
    ui.render_chart_card("Quy mô cụm K-Means", "Đối chiếu C0 và C1; mã cụm giữa hai cấu hình không tương ứng trực tiếp.", _safe_plot(count_fig), height=350)
    visual = analysis["visual"]
    pca_col1, pca_col2 = st.columns(2)
    with pca_col1:
        c1_fig = px.scatter(
            visual, x="PCA 1", y="PCA 2", color=visual["Cluster_C1"].astype(str),
            hover_data=["user_id", "Phân khúc gợi ý"], opacity=0.72,
            color_discrete_sequence=["#4355d8", "#16845b", "#d18a25", "#61738d", "#a05d99", "#2c8b9b"],
        )
        ui.render_chart_card("C1 · K-Means", "RFM + hành vi · PCA 2D để quan sát.", _safe_plot(c1_fig), height=390)
    with pca_col2:
        c2_fig = px.scatter(
            visual, x="PCA 1", y="PCA 2", color="Nhãn DBSCAN",
            hover_data=["user_id", "Cluster_C2"], opacity=0.72,
            color_discrete_sequence=["#4355d8", "#16845b", "#d18a25", "#9aa6b7", "#a05d99"],
        )
        ui.render_chart_card("C2 · DBSCAN", "Cùng mẫu trực quan PCA; chỉ khách đã chạy DBSCAN có nhãn.", _safe_plot(c2_fig), height=390)
    st.caption("PCA là phép chiếu để quan sát; các chỉ số benchmark được tính trong không gian đặc trưng đã chuẩn hóa.")

elif view_name == "Insights":
    ui.render_section_header("Gợi ý cần kiểm tra", "Tóm tắt dựa trên kết quả đã tính; các nhãn hành vi không phải dự báo hoặc quyết định tự động.")
    largest = profile.sort_values("So_khach_hang", ascending=False).iloc[0]
    buyer_rate = float(features["Purchase_Flag"].mean())
    insight_cols = st.columns(3)
    with insight_cols[0]:
        ui.render_metric_card("Quy mô lớn nhất", f"{int(largest['So_khach_hang']):,} khách", str(largest["Phân khúc gợi ý"]), "◉")
    with insight_cols[1]:
        ui.render_metric_card("Tỷ lệ từng mua", f"{buyer_rate:.1%}", "Trong tập khách hợp lệ sau làm sạch", "✓")
    with insight_cols[2]:
        ui.render_metric_card("Nhóm DBSCAN", f"{metadata['dbscan_sample_count']:,} khách", "Mẫu khách hợp lệ được chạy DBSCAN", "⌁")
    with st.container(border=True):
        st.markdown(f"**{largest['Phân khúc gợi ý']}** · {largest['Ty_le']:.1%} tổng số khách. {largest['Hướng hành động tham khảo']}")
        st.caption("Đây là hướng tham khảo dựa trên trung bình phân khúc. Cờ Recency không dự báo churn; CLV chưa được tính trong nghiên cứu.")

    with st.expander("Khám phá và xuất danh sách khách hàng", expanded=True):
        with st.container(horizontal=True):
            model_choice = st.selectbox(
                "Nhãn phân cụm", ["K-Means C1 · toàn bộ khách", "DBSCAN C2 · chỉ mẫu đã chạy"]
            )
            inactive_days = st.number_input("Ngưỡng Recency cảnh báo (ngày)", min_value=1, max_value=365, value=90)
            row_limit = st.selectbox("Số dòng xuất tối đa", [1_000, 5_000, 10_000, 50_000], index=2)
        search_id = st.text_input("Tìm theo mã khách hàng", placeholder="Ví dụ: demo_00042")
        only_inactive = st.toggle("Chỉ xem người mua có Recency vượt ngưỡng", value=False)
        if model_choice.startswith("K-Means"):
            available = profile.sort_values("Cluster_C1")["Cluster_C1"].astype(int).tolist()
            cluster_filter = st.selectbox(
                "Lọc phân khúc",
                ["Tất cả phân khúc"] + [
                    f"Cụm {cluster} · {profile.loc[profile['Cluster_C1'].eq(cluster), 'Phân khúc gợi ý'].iloc[0]}"
                    for cluster in available
                ],
            )
            customer_view = features.copy()
            customer_view["Nhãn cụm"] = customer_view["Cluster_C1"].map(lambda value: f"Cụm {int(value)}")
            if cluster_filter != "Tất cả phân khúc":
                selected_cluster = int(cluster_filter.split(" · ")[0].replace("Cụm ", ""))
                customer_view = customer_view.loc[customer_view["Cluster_C1"].eq(selected_cluster)]
        else:
            c2_values = sorted(int(value) for value in features["Cluster_C2"].unique() if value != -2)
            cluster_options = ["Tất cả mẫu DBSCAN", "Nhiễu (-1)"] + [f"Cụm {value}" for value in c2_values if value >= 0]
            cluster_filter = st.selectbox("Lọc nhãn DBSCAN", cluster_options)
            customer_view = features.loc[features["Cluster_C2"].ne(-2)].copy()
            customer_view["Nhãn cụm"] = customer_view["Cluster_C2"].map(
                lambda value: "Nhiễu (-1)" if value == -1 else f"Cụm {int(value)}"
            )
            if cluster_filter == "Nhiễu (-1)":
                customer_view = customer_view.loc[customer_view["Cluster_C2"].eq(-1)]
            elif cluster_filter.startswith("Cụm "):
                selected_cluster = int(cluster_filter.replace("Cụm ", ""))
                customer_view = customer_view.loc[customer_view["Cluster_C2"].eq(selected_cluster)]
        customer_view["Cờ theo dõi"] = np.where(
            customer_view["Frequency"].gt(0) & customer_view["Recency"].ge(inactive_days),
            "Người mua lâu chưa quay lại · heuristic",
            np.where(customer_view["Frequency"].eq(0), "Chưa từng mua", "Hoạt động gần đây"),
        )
        if only_inactive:
            customer_view = customer_view.loc[
                customer_view["Frequency"].gt(0) & customer_view["Recency"].ge(inactive_days)
            ]
        if search_id.strip():
            customer_view = customer_view.loc[
                customer_view["user_id"].astype(str).str.contains(search_id.strip(), case=False, na=False)
            ]
        st.caption(
            f"{len(customer_view):,} khách khớp bộ lọc. Cờ theo dõi là quy tắc Recency, không phải mô hình dự báo churn."
        )
        display_columns = [
            "user_id", "Nhãn cụm", "Phân khúc gợi ý", "Recency", "Frequency", "Monetary",
            "Purchase_Flag", "View_Count", "Cart_Count", "Cart_to_Purchase_Gap", "Cờ theo dõi",
        ]
        preview = customer_view[display_columns].head(int(row_limit))
        st.dataframe(
            preview,
            column_config={
                "user_id": st.column_config.TextColumn("Mã khách hàng"),
                "Nhãn cụm": st.column_config.TextColumn("Nhãn cụm"),
                "Phân khúc gợi ý": st.column_config.TextColumn("Diễn giải gợi ý"),
                "Recency": st.column_config.NumberColumn("Recency (ngày)", format="%.1f"),
                "Frequency": st.column_config.NumberColumn("Số lần mua", format="localized"),
                "Monetary": st.column_config.NumberColumn("Chi tiêu (USD)", format="$%.2f"),
                "Purchase_Flag": st.column_config.NumberColumn("Đã từng mua", format="localized"),
                "View_Count": st.column_config.NumberColumn("Lượt xem", format="localized"),
                "Cart_Count": st.column_config.NumberColumn("Lượt thêm giỏ", format="localized"),
                "Cart_to_Purchase_Gap": st.column_config.NumberColumn("Gap (xấp xỉ)", format="%.2f"),
            },
            hide_index=True,
            width="stretch",
            height=440,
        )
        export_data = preview.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            ":material/download: Tải CSV đã lọc",
            data=export_data,
            file_name="customer_segments_filtered.csv",
            mime="text/csv",
            type="primary",
        )
        if len(customer_view) > row_limit:
            st.warning(f"Tệp xuất giới hạn ở {row_limit:,} dòng đầu tiên để tránh tạo tệp quá lớn.")

st.markdown("---")
st.caption("CLV chưa được tính vì nghiên cứu chưa định nghĩa/cung cấp công thức kiểm chứng. Các tên phân khúc và hành động là gợi ý, không phải quyết định tự động.")
