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


st.set_page_config(
    page_title="Customer Behavior Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @keyframes riseIn {
      from { opacity: 0; transform: translateY(14px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes glowDrift {
      0%, 100% { transform: translate(0, 0) scale(1); opacity: .34; }
      50% { transform: translate(-20px, 12px) scale(1.12); opacity: .52; }
    }
    .hero-shell {
      position: relative; overflow: hidden; isolation: isolate;
      padding: 1.5rem 1.75rem; margin: .15rem 0 1.1rem;
      border: 1px solid rgba(126, 150, 255, .28); border-radius: 22px;
      background: linear-gradient(118deg, #111b3a 0%, #182a54 54%, #153e4b 100%);
      box-shadow: 0 18px 48px rgba(19, 37, 85, .18);
      animation: riseIn .65s ease-out both;
    }
    .hero-shell:after {
      content: ""; position: absolute; z-index: -1; width: 240px; height: 240px;
      right: 8%; top: -125px; border-radius: 50%;
      background: radial-gradient(circle, rgba(102, 221, 211, .8), rgba(102, 221, 211, 0));
      filter: blur(5px); animation: glowDrift 7s ease-in-out infinite;
    }
    .hero-kicker { color: #a9c3ff; font-size: .76rem; letter-spacing: .13em;
      text-transform: uppercase; font-weight: 700; margin-bottom: .55rem; }
    .hero-title { color: #f5f7ff; font-size: clamp(1.55rem, 3vw, 2.35rem);
      line-height: 1.12; font-weight: 750; margin: 0 0 .55rem; }
    .hero-copy { color: #c5d1ef; max-width: 850px; font-size: .98rem; margin: 0; }
    .hero-pill { display: inline-block; margin-top: .95rem; padding: .33rem .7rem;
      border: 1px solid rgba(188, 211, 255, .25); border-radius: 999px;
      color: #d8e5ff; background: rgba(255,255,255,.08); font-size: .77rem; }
    @media (prefers-reduced-motion: reduce) {
      .hero-shell, .hero-shell:after { animation: none !important; }
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
        font=dict(family="Arial, sans-serif", color="#26324b"),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


st.markdown(
    """
    <section class="hero-shell">
      <div class="hero-kicker">BIT Genesis Research Awards 2026 · Customer analytics</div>
      <h1 class="hero-title">Hành vi khách hàng, nhìn rõ hơn</h1>
      <p class="hero-copy">Từ clickstream và giao dịch đến phân khúc có thể diễn giải —
      so sánh RFM với bộ đặc trưng hành vi trong một dashboard tương tác.</p>
      <span class="hero-pill">RFM · K-Means · DBSCAN · Benchmark</span>
    </section>
    """,
    unsafe_allow_html=True,
)

st.caption("Prototype chạy cục bộ: ETL bằng Pandas hoặc Spark local, phân cụm bằng scikit-learn. Kết quả từ dữ liệu lịch sử, không phải dashboard real-time.")

with st.sidebar:
    st.header(":material/tune: Thiết lập phân tích")
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
            ":material/play_arrow: Chạy pipeline",
            type="primary",
            width="stretch",
        )

    st.divider()
    if source_name == "Dữ liệu minh họa tổng hợp":
        st.caption("Demo: 50.000 sự kiện → 2.000 người dùng ban đầu → tối đa 1.700 khách hợp lệ sau lọc.")
    else:
        st.caption("Cấu hình theo luận văn: k = 4; DBSCAN lấy mẫu tối đa 50.000 khách hàng hợp lệ.")

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
if analysis is None:
    st.info("Chọn dữ liệu và tham số ở thanh bên, sau đó chạy pipeline để mở dashboard.", icon=":material/insights:")
    st.stop()

features = analysis["features"]
benchmark = analysis["benchmark"]
profile = analysis["profile"]
metadata = analysis["metadata"]
event_counts = analysis["event_counts"]

with st.container(border=True):
    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"**Nguồn:** {metadata['source']}  ·  **ETL:** {metadata['engine']}  ·  **Chạy gần nhất:** {metadata['run_at']}")
        st.caption(
            f"Khoảng thời gian dữ liệu: {metadata['period_start']:%d/%m/%Y} – "
            f"{metadata['period_end']:%d/%m/%Y}  ·  DBSCAN chạy trên mẫu "
            f"{metadata['dbscan_sample_count']:,} khách hàng"
        )
    with right:
        st.badge("Minh họa tổng hợp" if "minh họa" in metadata["source"].lower() else "CSV đã tải", color="blue")

if metadata["unknown_event_types"]:
    st.warning("Có loại sự kiện ngoài view/cart/remove_from_cart/purchase: " + ", ".join(metadata["unknown_event_types"]))

view_name = st.segmented_control(
    "Khu vực dashboard",
    ["Tổng quan", "Benchmark", "Chân dung cụm", "Khách hàng"],
    default="Tổng quan",
    key="dashboard_section",
    label_visibility="collapsed",
)

if view_name == "Tổng quan":
    st.subheader(":material/dashboard: Tổng quan")
    buyer_rate = float(features["Purchase_Flag"].mean())
    mean_revenue = float(features["Monetary"].mean())
    with st.container(horizontal=True):
        st.metric("Khách hàng được phân tích", f"{len(features):,}", border=True)
        st.metric("Sự kiện sau làm sạch", f"{metadata['clean_event_count']:,}", border=True)
        st.metric("Khách từng mua", f"{buyer_rate:.1%}", border=True)
        st.metric("Chi tiêu trung bình / khách", _format_money(mean_revenue), border=True)

    chart_col1, chart_col2 = st.columns([1.05, 1.4])
    with chart_col1:
        st.markdown("#### Cơ cấu sự kiện")
        event_chart_data = event_counts.copy()
        event_chart_data["Tỷ trọng"] = event_chart_data["Số sự kiện"] / max(1, event_chart_data["Số sự kiện"].sum())
        event_chart_data = event_chart_data.sort_values("Số sự kiện")
        event_fig = px.bar(
            event_chart_data,
            x="Số sự kiện",
            y="Loại sự kiện",
            orientation="h",
            color="Loại sự kiện",
            text=event_chart_data["Tỷ trọng"].map(lambda v: f"{v:.1%}"),
            title="Tỷ trọng theo loại hành vi · không phải tỷ lệ chuyển đổi",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        event_fig.update_layout(showlegend=False, xaxis_title="Số sự kiện", yaxis_title="")
        st.plotly_chart(_safe_plot(event_fig), width="stretch", config={"displayModeBar": False})

    with chart_col2:
        st.markdown("#### Quy mô các phân khúc K-Means")
        size_fig = px.bar(
            profile.sort_values("So_khach_hang", ascending=False),
            x="Phân khúc gợi ý",
            y="So_khach_hang",
            color="Phân khúc gợi ý",
            text="So_khach_hang",
            title="Số khách và tỷ trọng theo cụm C1",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        size_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Số khách hàng")
        st.plotly_chart(_safe_plot(size_fig), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Không gian RFM · mẫu hiển thị tối đa 3.000 khách")
    rfm_sample = analysis["visual"]
    rfm_fig = px.scatter_3d(
        rfm_sample,
        x="Recency",
        y="Frequency",
        z="Monetary",
        color="Phân khúc gợi ý",
        hover_data=["user_id", "Cluster_C1", "Cart_Count"],
        title="Mỗi điểm là một khách hàng; màu là nhãn diễn giải gợi ý",
        opacity=0.76,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    rfm_fig.update_layout(height=570, scene=dict(
        xaxis_title="Recency · ngày", yaxis_title="Frequency · số lần mua", zaxis_title="Monetary · USD"
    ))
    st.plotly_chart(_safe_plot(rfm_fig), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Phễu làm sạch · tách biệt số sự kiện và số khách hàng")
    funnel = analysis["funnel"].copy()
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

elif view_name == "Benchmark":
    st.subheader(":material/compare_arrows: So sánh ba cấu hình thực nghiệm")
    st.caption("C0–C1 thay đổi bộ đặc trưng; C1–C2 giữ bộ đặc trưng và đổi thuật toán. C0/C1 chấm trên tối đa 5.000 khách; DBSCAN chạy trên mẫu tối đa 50.000 khách.")
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
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.info("Silhouette cao hơn thường tốt hơn. C0/C1 được chấm trên tối đa 5.000 khách; C2 bỏ nhiễu trước khi chấm.")
    metric_col2.info("Davies–Bouldin thấp hơn thường tốt hơn; Calinski–Harabasz cao hơn thường tốt hơn.")
    metric_col3.warning("Không chọn mô hình chỉ bằng một chỉ số. DBSCAN chỉ có nhãn cho mẫu đã chạy.")

    count_rows = []
    for config_name, cluster_column in [("C0 · RFM", "Cluster_C0"), ("C1 · RFM + hành vi", "Cluster_C1")]:
        counts = features[cluster_column].value_counts().sort_index()
        count_rows.extend(
            {"Cấu hình": config_name, "Cụm": f"Cụm {int(cluster)}", "Số khách": int(count)}
            for cluster, count in counts.items()
        )
    count_fig = px.bar(
        pd.DataFrame(count_rows),
        x="Cụm",
        y="Số khách",
        color="Cấu hình",
        barmode="group",
        title="Quy mô phân cụm · đối chiếu C0 và C1",
        color_discrete_sequence=["#6574cd", "#22a699"],
    )
    st.plotly_chart(_safe_plot(count_fig), width="stretch", config={"displayModeBar": False})
    st.caption("Mã cụm K-Means chỉ là nhãn kỹ thuật; Cụm 0 ở C0 không nhất thiết tương ứng với Cụm 0 ở C1.")

    visual = analysis["visual"]
    pca_col1, pca_col2 = st.columns(2)
    with pca_col1:
        c1_fig = px.scatter(
            visual,
            x="PCA 1",
            y="PCA 2",
            color=visual["Cluster_C1"].astype(str),
            hover_data=["user_id", "Phân khúc gợi ý"],
            title="C1 · K-Means trên đặc trưng đề xuất",
            opacity=0.72,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        st.plotly_chart(_safe_plot(c1_fig), width="stretch", config={"displayModeBar": False})
    with pca_col2:
        c2_fig = px.scatter(
            visual,
            x="PCA 1",
            y="PCA 2",
            color="Nhãn DBSCAN",
            hover_data=["user_id", "Cluster_C2"],
            title="C2 · DBSCAN trên cùng mẫu PCA",
            opacity=0.72,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        st.plotly_chart(_safe_plot(c2_fig), width="stretch", config={"displayModeBar": False})
    st.caption("PCA chỉ là phép chiếu 2D để quan sát; các chỉ số benchmark được tính trong không gian đặc trưng đã chuẩn hóa.")

elif view_name == "Chân dung cụm":
    st.subheader(":material/groups: Chân dung khách hàng theo C1")
    st.caption("Tên phân khúc và hướng hành động bên dưới là gợi ý diễn giải theo trung bình cụm, cần chuyên gia rà soát trước khi triển khai tiếp thị.")
    profile_view = profile.rename(columns={
        "Cluster_C1": "Cụm C1",
        "So_khach_hang": "Số khách hàng",
        "Ty_le": "Tỷ trọng khách",
        "Recency_tb": "Recency TB (ngày)",
        "Frequency_tb": "Frequency TB (lần mua)",
        "Monetary_tb": "Monetary TB (USD)",
        "Ti_le_da_mua": "Tỷ lệ từng mua",
        "View_tb": "Lượt xem TB",
        "Cart_tb": "Lượt thêm giỏ TB",
        "Gap_tb": "Cart-to-purchase gap TB",
        "Span_tb": "Chu kỳ tương tác TB (ngày)",
        "Session_tb": "Phiên TB",
    })
    st.dataframe(
        profile_view,
        column_config={
            "Tỷ trọng khách": st.column_config.NumberColumn(format="percent"),
            "Tỷ lệ từng mua": st.column_config.NumberColumn(format="percent"),
            "Monetary TB (USD)": st.column_config.NumberColumn(format="$%.2f"),
            "Recency TB (ngày)": st.column_config.NumberColumn(format="%.1f"),
            "Frequency TB (lần mua)": st.column_config.NumberColumn(format="%.2f"),
            "Hướng hành động tham khảo": st.column_config.TextColumn(width="large"),
        },
        hide_index=True,
        width="stretch",
    )

    radar_columns = [
        "Recency_tb", "Frequency_tb", "Monetary_tb", "View_tb", "Cart_tb", "Span_tb", "Session_tb"
    ]
    radar_labels = ["Recency", "Frequency", "Monetary", "Views", "Cart", "Span", "Sessions"]
    denominator = (profile[radar_columns].max() - profile[radar_columns].min()).replace(0, 1)
    normalized = (profile[radar_columns] - profile[radar_columns].min()) / denominator
    radar_fig = go.Figure()
    for idx, row in profile.iterrows():
        values = normalized.loc[idx].tolist()
        radar_fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_labels + [radar_labels[0]],
            name=f"Cụm {int(row['Cluster_C1'])} · {row['Phân khúc gợi ý']}",
            fill="toself",
            opacity=0.35,
        ))
    radar_fig.update_layout(
        title="Radar đặc trưng cụm · chuẩn hóa min–max để so sánh tương đối",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        height=560,
    )
    st.plotly_chart(_safe_plot(radar_fig), width="stretch", config={"displayModeBar": False})

elif view_name == "Khách hàng":
    st.subheader(":material/manage_search: Khám phá và xuất danh sách khách hàng")
    with st.container(horizontal=True):
        model_choice = st.selectbox(
            "Nhãn phân cụm",
            ["K-Means C1 · toàn bộ khách", "DBSCAN C2 · chỉ mẫu đã chạy"],
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
        f"{len(customer_view):,} khách khớp bộ lọc. Cờ theo dõi là quy tắc Recency, "
        "không phải mô hình dự báo churn."
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

st.divider()
st.caption("CLV chưa được tính vì nghiên cứu chưa định nghĩa/cung cấp công thức kiểm chứng. Các tên phân khúc và hành động là gợi ý, không phải quyết định tự động.")
