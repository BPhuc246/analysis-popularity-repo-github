"""
GitHub Repository Popularity — Modern Interactive Dashboard
Run:
    streamlit run dashboard.py

Optional:
    set GITHUB_REPO_DATA_DIR to the folder containing the CSV files.
    Required raw file prefix: 01_github_repos_chua_chuan_hoa_raw
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Page + theme
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="GitHub Repository Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RAW_PREFIX = "01_github_repos_chua_chuan_hoa_raw"
NORMALIZED_PREFIXES = {
    "Min-Max": "minmax_github_repos_binned",
    "Scale": "scale_github_repos_binned",
    "Z-Score": "zscore_github_repos_binned",
}

COLORS = {
    "bg": "#0B1020",
    "panel": "#111A2E",
    "panel_2": "#151F35",
    "border": "#25334D",
    "text": "#E8EEF9",
    "muted": "#91A0B8",
    "blue": "#5B8CFF",
    "cyan": "#42C6D9",
    "purple": "#A78BFA",
    "green": "#39C69B",
    "orange": "#F4A261",
    "pink": "#F472B6",
}

PLOT_TEMPLATE = "plotly_dark"
px.defaults.template = PLOT_TEMPLATE
px.defaults.color_discrete_sequence = [
    COLORS["blue"], COLORS["cyan"], COLORS["purple"],
    COLORS["green"], COLORS["orange"], COLORS["pink"],
]

# Lucide icon SVGs (Lucide-compatible stroke style; no external package/CDN).
LUCIDE = {
    "github": '<path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>',
    "star": '<path d="m12 3 2.7 5.47 6.03.88-4.36 4.25 1.03 6.01L12 16.77l-5.4 2.84 1.03-6.01-4.36-4.25 6.03-.88L12 3z"/>',
    "git-fork": '<circle cx="6" cy="18" r="3"/><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><path d="M6 9v6a3 3 0 0 0 3 3h3"/><path d="M18 9a9 9 0 0 1-9 9"/>',
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/><line x1="14" x2="10" y1="4" y2="20"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "layers": '<path d="m12 2 9 5-9 5-9-5 9-5z"/><path d="m3 12 9 5 9-5"/><path d="m3 17 9 5 9-5"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "book": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
}

def icon(name: str, color: str = "#91A0B8", size: int = 18) -> str:
    paths = LUCIDE.get(name, LUCIDE["chart"])
    return (
        f'<span class="lucide-wrap"><svg width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths}</svg></span>'
    )

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --bg:#0B1020; --panel:#111A2E; --panel2:#151F35; --line:#25334D;
            --text:#E8EEF9; --muted:#91A0B8; --blue:#5B8CFF; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background: radial-gradient(ellipse at 12% 0%, #172647 0%, #0B1020 42%, #090E1B 100%); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(11,16,32,.78); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#10192C 0%,#0D1425 100%); border-right:1px solid #25334D; }
    [data-testid="stSidebar"] > div { padding-top:1.3rem; }
    .block-container { padding-top:2rem; padding-bottom:3rem; max-width:1500px; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.035em; color:#F2F6FF; }
    h1 { font-size:2.15rem !important; }
    h2 { font-size:1.42rem !important; }
    h3 { font-size:1.08rem !important; }
    p, label, .stMarkdown { color:#C6D0E1; }
    .hero { padding:1.35rem 1.55rem; border:1px solid #2A3A58; border-radius:20px;
            background:linear-gradient(115deg,rgba(36,64,112,.48),rgba(17,26,46,.92) 62%,rgba(29,45,74,.72));
            box-shadow:0 16px 50px rgba(0,0,0,.18); margin-bottom:1.1rem; }
    .hero-kicker { color:#91B6FF; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
    .hero-title { font-family:'Space Grotesk',sans-serif; color:#F5F8FF; font-size:1.9rem; font-weight:700; margin:.3rem 0; }
    .hero-sub { color:#AAB8CE; font-size:.92rem; }
    .hero-chip { display:inline-flex; align-items:center; gap:.4rem; border:1px solid #38527B; border-radius:999px;
                 padding:.32rem .68rem; color:#C6D8FF; background:rgba(91,140,255,.10); font-size:.75rem; margin-top:.75rem; }
    .kpi { background:linear-gradient(145deg,rgba(21,31,53,.98),rgba(15,24,42,.98)); border:1px solid #263752;
           border-radius:16px; padding:1rem 1.05rem; min-height:126px; box-shadow:0 8px 25px rgba(0,0,0,.12); }
    .kpi-top { display:flex; align-items:center; justify-content:space-between; color:#A5B4CA; font-size:.78rem; font-weight:600; }
    .kpi-icon { width:34px;height:34px;border-radius:11px;display:flex;align-items:center;justify-content:center;background:rgba(91,140,255,.13); }
    .kpi-value { color:#F4F7FF; font-family:'Space Grotesk',sans-serif; font-size:1.65rem; font-weight:700; margin-top:.72rem; line-height:1.1; }
    .kpi-note { color:#7F91AD; font-size:.72rem; margin-top:.35rem; }
    .section-label { color:#91B6FF; text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; font-weight:700; margin-bottom:.3rem; }
    .panel-note { color:#8292AC; font-size:.78rem; margin-top:-.35rem; }
    .stTabs [data-baseweb="tab-list"] { gap:.45rem; border-bottom:1px solid #263752; }
    .stTabs [data-baseweb="tab"] { background:#111A2E; border:1px solid #25334D; border-radius:10px 10px 0 0; padding:.65rem 1rem; color:#AAB8CE; }
    .stTabs [aria-selected="true"] { background:#1B2B49 !important; color:#F4F7FF !important; border-color:#4569A5 !important; }
    .stButton button, .stDownloadButton button { border-radius:10px; border:1px solid #36517D; background:#1B2B49; color:#EAF1FF; }
    .stButton button:hover, .stDownloadButton button:hover { border-color:#6B98F4; color:white; }
    div[data-testid="stMetric"] { background:#111A2E; border:1px solid #263752; border-radius:14px; padding:1rem; }
    div[data-testid="stMetricLabel"] { color:#AAB8CE; }
    div[data-testid="stMetricValue"] { color:#F4F7FF; font-family:'Space Grotesk',sans-serif; }
    [data-testid="stDataFrame"] { border:1px solid #263752; border-radius:12px; overflow:hidden; }
    .lucide-wrap { display:inline-flex; vertical-align:middle; }
    .sidebar-brand { display:flex; gap:.65rem; align-items:center; padding:.25rem .2rem 1.2rem; color:#F4F7FF; font-family:'Space Grotesk'; font-weight:700; font-size:1.05rem; }
    .sidebar-caption { color:#7F91AD; font-size:.72rem; margin-top:-.9rem; margin-bottom:1.2rem; }
    .footer { color:#74849E; font-size:.75rem; border-top:1px solid #263752; padding-top:1rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def find_data_dir() -> Path:
    candidates = []
    env = os.getenv("GITHUB_REPO_DATA_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [ROOT / "data" / "input", ROOT / "data", ROOT, Path.cwd()]
    for folder in candidates:
        folder = folder.resolve()
        if folder.exists() and list(folder.glob(f"{RAW_PREFIX}*.csv")):
            return folder
    raise FileNotFoundError(
        f"Không tìm thấy CSV bắt đầu bằng '{RAW_PREFIX}'. "
        "Đặt file trong data/input/ hoặc cấu hình GITHUB_REPO_DATA_DIR."
    )

def find_csv(folder: Path, prefix: str) -> Path | None:
    matches = sorted(folder.glob(f"{prefix}*.csv"))
    return matches[0] if matches else None

def parse_json_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []

def prepare_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in ["created_at", "updated_at", "pushed_at", "collected_at"]:
        if col in df:
            df[f"{col}_dt"] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if {"created_at_dt", "collected_at_dt"}.issubset(df.columns):
        df["age_days"] = (df["collected_at_dt"] - df["created_at_dt"]).dt.total_seconds() / 86400
        df["age_years"] = df["age_days"] / 365.25
        df["created_year"] = df["created_at_dt"].dt.year.astype("Int64")
    df["topics_list"] = df["topics"].map(parse_json_list) if "topics" in df else [[] for _ in range(len(df))]
    df["topic_count"] = df["topics_list"].map(lambda xs: len({str(x).strip().casefold() for x in xs if str(x).strip()}))
    if "primary_language" in df:
        df["primary_language_clean"] = df["primary_language"].astype("string").str.strip()
        df["primary_language_key"] = df["primary_language_clean"].str.casefold()
    else:
        df["primary_language_clean"] = "Unknown"
        df["primary_language_key"] = "unknown"
    return df

@st.cache_data(show_spinner="Đang tải dữ liệu GitHub...")
def load_data(folder_text: str):
    folder = Path(folder_text)
    raw_path = find_csv(folder, RAW_PREFIX)
    if raw_path is None:
        raise FileNotFoundError(f"Không tìm thấy file raw trong {folder}")
    raw = pd.read_csv(raw_path, low_memory=False)
    normalized = {}
    for method, prefix in NORMALIZED_PREFIXES.items():
        path = find_csv(folder, prefix)
        if path:
            normalized[method] = pd.read_csv(path, low_memory=False)
    return raw, prepare_frame(raw), normalized, raw_path.name

try:
    DATA_DIR = find_data_dir()
    raw, base_df, normalized, raw_filename = load_data(str(DATA_DIR))
except Exception as exc:
    st.error(f"Không thể nạp dữ liệu: {exc}")
    st.info("Kiểm tra thư mục CSV và biến môi trường GITHUB_REPO_DATA_DIR.")
    st.stop()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def fmt_num(value, decimals=0):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}"

def language_frame(df):
    if "primary_language_key" not in df:
        return df.iloc[0:0]
    mask = df["primary_language_key"].notna() & df["primary_language_key"].ne("") & df["primary_language_key"].ne("text")
    return df.loc[mask].copy()

def topic_counts(df):
    counts = Counter()
    for values in df.get("topics_list", pd.Series([[]] * len(df))):
        counts.update({str(v).strip().casefold() for v in values if str(v).strip()})
    return pd.Series(dict(counts), dtype="int64").sort_values(ascending=False)

def style_fig(fig, height=390):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#C6D0E1", size=12),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=16, color="#F0F5FF"), x=0.02),
        margin=dict(l=18, r=18, t=58, b=22),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.08, x=1, xanchor="right"),
        hoverlabel=dict(bgcolor="#18243A", bordercolor="#40577C", font=dict(color="#F4F7FF")),
    )
    fig.update_xaxes(gridcolor="rgba(145,160,184,.12)", zerolinecolor="rgba(145,160,184,.15)")
    fig.update_yaxes(gridcolor="rgba(145,160,184,.12)", zerolinecolor="rgba(145,160,184,.15)")
    return fig

def show_chart(fig, height=390):
    st.plotly_chart(style_fig(fig, height), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

def kpi_card(label, value, note, icon_name, accent):
    st.markdown(
        f"""
        <div class="kpi">
          <div class="kpi-top"><span>{label}</span><span class="kpi-icon">{icon(icon_name, accent, 18)}</span></div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------
st.sidebar.markdown(
    f'<div class="sidebar-brand">{icon("github", "#F4F7FF", 25)} GitHub Analytics</div>'
    '<div class="sidebar-caption">REPOSITORY POPULARITY EXPLORER</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(f'**{icon("filter", "#91B6FF")} Bộ lọc dữ liệu**', unsafe_allow_html=True)

search_text = st.sidebar.text_input("Tìm repository", placeholder="Tên repo hoặc owner...")
lf_all = language_frame(base_df)
language_options = sorted(lf_all["primary_language_clean"].dropna().unique().tolist())
selected_languages = st.sidebar.multiselect("Ngôn ngữ chính", language_options, default=[])
if "star_count" in base_df:
    star_min = int(max(0, np.nanmin(base_df["star_count"])))
    star_max = int(np.nanmax(base_df["star_count"]))
    if star_min == star_max:
        star_range = (star_min, star_max)
        st.sidebar.caption(f"Stars: {star_min:,}")
    else:
        star_range = st.sidebar.slider("Khoảng Stars", star_min, star_max, (star_min, star_max), format="%d")
else:
    star_range = (0, 0)

issue_filter = st.sidebar.selectbox("Issue tracker", ["Tất cả", "Có Issues", "Không có Issues"])
wiki_filter = st.sidebar.selectbox("Wiki", ["Tất cả", "Có Wiki", "Không có Wiki"])
min_forks = st.sidebar.number_input("Forks tối thiểu", min_value=0, value=0, step=10)
st.sidebar.markdown("---")
st.sidebar.caption(f"📁 Nguồn: `{raw_filename}`")
st.sidebar.caption(f"🧾 {len(base_df):,} repositories trong dữ liệu gốc")

filtered = base_df.copy()
if search_text.strip():
    term = search_text.strip().casefold()
    searchable = filtered.get("full_name", filtered.get("repo_name", pd.Series("", index=filtered.index))).astype("string").str.casefold()
    filtered = filtered[searchable.str.contains(term, na=False)]
if selected_languages:
    filtered = filtered[filtered["primary_language_clean"].isin(selected_languages)]
if "star_count" in filtered:
    filtered = filtered[filtered["star_count"].between(star_range[0], star_range[1])]
if "forks_count" in filtered:
    filtered = filtered[filtered["forks_count"].fillna(0) >= min_forks]
if issue_filter == "Có Issues" and "has_issues" in filtered:
    filtered = filtered[filtered["has_issues"].eq(True)]
elif issue_filter == "Không có Issues" and "has_issues" in filtered:
    filtered = filtered[filtered["has_issues"].eq(False)]
if wiki_filter == "Có Wiki" and "has_wiki" in filtered:
    filtered = filtered[filtered["has_wiki"].eq(True)]
elif wiki_filter == "Không có Wiki" and "has_wiki" in filtered:
    filtered = filtered[filtered["has_wiki"].eq(False)]

# ---------------------------------------------------------------------
# Header + KPI
# ---------------------------------------------------------------------
snapshot = "Snapshot không xác định"
if "collected_at_dt" in base_df and base_df["collected_at_dt"].notna().any():
    snapshot = base_df["collected_at_dt"].dropna().max().strftime("%d %b %Y · %H:%M UTC")

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">DATA ANALYTICS · GITHUB ECOSYSTEM</div>
      <div class="hero-title">Repository Popularity Overview</div>
      <div class="hero-sub">Khám phá mức độ phổ biến của repositories qua Stars, Forks, ngôn ngữ lập trình và vòng đời dự án.</div>
      <div class="hero-chip">{icon("activity", "#91B6FF", 14)} {snapshot}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="section-label">EXECUTIVE SUMMARY</div>', unsafe_allow_html=True)

if filtered.empty:
    st.warning("Không có repository phù hợp với bộ lọc hiện tại. Hãy nới lỏng một hoặc nhiều điều kiện.")
    st.stop()

lang_filtered = language_frame(filtered)
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Repositories", fmt_num(len(filtered)), f"{len(filtered)/max(len(base_df),1):.1%} của mẫu gốc", "github", COLORS["blue"])
with k2:
    kpi_card("Median Stars", fmt_num(filtered["star_count"].median()), "Mức Stars trung vị", "star", COLORS["orange"])
with k3:
    kpi_card("Median Forks", fmt_num(filtered["forks_count"].median()), "Mức Forks trung vị", "git-fork", COLORS["cyan"])
with k4:
    kpi_card("Programming languages", fmt_num(lang_filtered["primary_language_clean"].nunique()), "Không tính nhãn text / trống", "code", COLORS["purple"])
with k5:
    age_med = filtered["age_years"].median() if "age_years" in filtered else np.nan
    kpi_card("Median repository age", f"{fmt_num(age_med, 1)} yrs", "Tính từ created_at đến collected_at", "calendar", COLORS["green"])

if "watchers_count" not in base_df.columns:
    st.caption("ⓘ Schema không có `watchers_count`; dashboard không hiển thị Watchers và không gộp Stars/Forks thành một điểm popularity.")

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------
tabs = st.tabs([
    "✦ Tổng quan",
    "⌘ Ngôn ngữ & Topics",
    "↗ Popularity",
    "◷ Tuổi & Tương quan",
    "▤ Chuẩn hóa",
    "☷ Repository Explorer",
])

with tabs[0]:
    st.markdown("## Bức tranh tổng quan")
    st.markdown('<div class="panel-note">Cấu trúc mẫu sau khi áp dụng bộ lọc hiện tại.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        top_lang = lang_filtered["primary_language_clean"].value_counts().head(10).rename_axis("Language").reset_index(name="Repositories")
        fig = px.bar(top_lang.sort_values("Repositories"), x="Repositories", y="Language", orientation="h",
                     title="Top 10 ngôn ngữ lập trình", color="Repositories",
                     color_continuous_scale=["#263D66", "#5B8CFF"])
        fig.update_layout(coloraxis_showscale=False)
        show_chart(fig, 420)
    with c2:
        bool_col = "has_issues"
        if bool_col in filtered:
            issue_data = filtered[bool_col].map({True: "Có Issues", False: "Không có Issues"}).fillna("Missing").value_counts().rename_axis("Trạng thái").reset_index(name="Repositories")
            fig = px.pie(issue_data, names="Trạng thái", values="Repositories", hole=.68,
                         title="Issue tracker", color="Trạng thái",
                         color_discrete_sequence=[COLORS["blue"], COLORS["cyan"], "#66758E"])
            fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#111A2E", width=3)))
            show_chart(fig, 420)
        else:
            st.info("Không có cột has_issues trong schema.")
    c3, c4 = st.columns([1.15, 1])
    with c3:
        if "created_year" in filtered:
            year_counts = filtered.dropna(subset=["created_year"]).groupby("created_year").size().rename("Repositories").reset_index()
            fig = px.area(year_counts, x="created_year", y="Repositories", markers=True, title="Repositories theo năm tạo")
            fig.update_traces(line=dict(color=COLORS["cyan"], width=2.5), fillcolor="rgba(66,198,217,.12)")
            show_chart(fig, 350)
    with c4:
        if "topic_count" in filtered:
            fig = px.histogram(filtered, x="topic_count", nbins=25, title="Số topics trên mỗi repository",
                               color_discrete_sequence=[COLORS["purple"]])
            show_chart(fig, 350)
    st.caption("Lưu ý: các năm đầu/cuối trong snapshot có thể không phải năm trọn vẹn. Đây là cơ cấu của mẫu dữ liệu, không đại diện toàn bộ GitHub.")

with tabs[1]:
    st.markdown("## Ngôn ngữ & Topics")
    a, b = st.columns(2)
    with a:
        lang_counts = lang_filtered["primary_language_clean"].value_counts().head(15).rename_axis("Language").reset_index(name="Repositories")
        fig = px.bar(lang_counts.sort_values("Repositories"), x="Repositories", y="Language", orientation="h",
                     title="Top 15 primary languages", color="Repositories",
                     color_continuous_scale=["#263D66", "#5B8CFF"])
        fig.update_layout(coloraxis_showscale=False)
        show_chart(fig, 470)
    with b:
        top5 = lang_filtered["primary_language_clean"].value_counts().head(5).index.tolist()
        box = lang_filtered[lang_filtered["primary_language_clean"].isin(top5)].copy()
        box["log1p_stars"] = np.log1p(box["star_count"].clip(lower=0))
        fig = px.box(box, x="primary_language_clean", y="log1p_stars", color="primary_language_clean",
                     points=False, title="Phân phối log(1 + Stars) · Top 5 languages")
        fig.update_layout(showlegend=False, xaxis_title="Primary language", yaxis_title="log(1 + Stars)")
        show_chart(fig, 470)
    st.markdown("### Topics nổi bật")
    topics = topic_counts(filtered).head(18).rename("Repositories").rename_axis("Topic").reset_index()
    if not topics.empty:
        fig = px.bar(topics.sort_values("Repositories"), x="Repositories", y="Topic", orientation="h",
                     title="Top 18 topics theo số repository", color="Repositories",
                     color_continuous_scale=["#243B62", "#A78BFA"])
        fig.update_layout(coloraxis_showscale=False)
        show_chart(fig, 510)
    else:
        st.info("Không có dữ liệu topics hợp lệ trong bộ lọc này.")

with tabs[2]:
    st.markdown("## Popularity analysis")
    st.markdown('<div class="panel-note">Stars và Forks được phân tích như hai chỉ số riêng biệt; log1p chỉ dùng để trực quan hóa phân phối lệch.</div>', unsafe_allow_html=True)
    metric_labels = {
        "star_count": "Stars",
        "forks_count": "Forks",
        "open_issues_count": "Open issues",
        "size_kb": "Repository size (KB)",
    }
    available_metrics = [m for m in metric_labels if m in filtered]
    chosen_metric = st.selectbox("Chỉ số phân phối", available_metrics, format_func=lambda x: metric_labels[x])
    metric_values = pd.to_numeric(filtered[chosen_metric], errors="coerce").dropna().clip(lower=0)
    fig = px.histogram(x=np.log1p(metric_values), nbins=48, title=f"Phân phối log(1 + {metric_labels[chosen_metric]})",
                       labels={"x": f"log(1 + {metric_labels[chosen_metric]})"},
                       color_discrete_sequence=[COLORS["blue"]])
    show_chart(fig, 370)
    sample_cols = [c for c in ["star_count", "forks_count", "has_wiki", "full_name"] if c in filtered]
    sample = filtered[sample_cols].dropna(subset=["star_count", "forks_count"]).sample(
        n=min(6000, filtered[["star_count", "forks_count"]].dropna().shape[0]), random_state=42
    ).copy()
    if not sample.empty:
        sample["log1p_stars"] = np.log1p(sample["star_count"].clip(lower=0))
        sample["log1p_forks"] = np.log1p(sample["forks_count"].clip(lower=0))
        fig = px.scatter(sample, x="log1p_stars", y="log1p_forks",
                         color="has_wiki" if "has_wiki" in sample else None,
                         hover_name="full_name" if "full_name" in sample else None,
                         opacity=.55, title="Mối liên hệ giữa Stars và Forks",
                         labels={"log1p_stars": "log(1 + Stars)", "log1p_forks": "log(1 + Forks)"})
        show_chart(fig, 470)
    st.caption("Không diễn giải Stars + Forks thành một điểm popularity tổng hợp. Mối liên hệ quan sát được không hàm ý quan hệ nhân quả.")

with tabs[3]:
    st.markdown("## Repository age & correlation")
    c1, c2 = st.columns([1, 1.1])
    with c1:
        if "age_years" in filtered:
            age_data = filtered["age_years"].dropna()
            fig = px.histogram(x=age_data, nbins=36, title="Phân phối tuổi repository",
                               labels={"x": "Tuổi repository (năm)"},
                               color_discrete_sequence=[COLORS["green"]])
            show_chart(fig, 410)
            st.metric("Tuổi trung vị", f"{age_data.median():.2f} năm")
        else:
            st.info("Không đủ cột ngày tháng để tính tuổi repository.")
    with c2:
        corr_cols = [c for c in ["star_count", "forks_count", "open_issues_count", "size_kb", "age_days", "topic_count"] if c in filtered]
        corr_method = st.radio("Phương pháp tương quan", ["spearman", "pearson"], horizontal=True)
        corr = filtered[corr_cols].apply(pd.to_numeric, errors="coerce").corr(method=corr_method)
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                        title=f"Correlation matrix · {corr_method.title()}", aspect="auto")
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        show_chart(fig, 470)
    st.caption("Tương quan không chứng minh quan hệ nhân quả. Với cỡ mẫu lớn, cần xem cả độ lớn hệ số và ý nghĩa thực tiễn.")

with tabs[4]:
    st.markdown("## So sánh dữ liệu chuẩn hóa")
    st.markdown('<div class="panel-note">Đối chiếu phân phối raw với các file chuẩn hóa được cung cấp. Missing values không được tự động impute.</div>', unsafe_allow_html=True)
    if not normalized:
        st.warning("Không tìm thấy file chuẩn hóa. Hãy đặt các CSV Min-Max / Scale / Z-Score cùng thư mục dữ liệu raw.")
    else:
        norm_metric = st.selectbox("Biến số", [m for m in ["star_count", "forks_count", "open_issues_count", "size_kb"] if m in raw],
                                   format_func=lambda x: {"star_count":"Stars","forks_count":"Forks","open_issues_count":"Open issues","size_kb":"Size (KB)"}[x])
        norm_method = st.selectbox("Phương pháp / file", list(normalized.keys()))
        norm_df = normalized[norm_method]
        if norm_metric not in norm_df:
            st.error(f"File {norm_method} không có cột {norm_metric}.")
        else:
            raw_series = pd.to_numeric(raw[norm_metric], errors="coerce").dropna()
            norm_series = pd.to_numeric(norm_df[norm_metric], errors="coerce").dropna()
            stat = pd.DataFrame({
                "Raw": raw_series.describe(percentiles=[.25, .5, .75]),
                norm_method: norm_series.describe(percentiles=[.25, .5, .75]),
            })
            st.dataframe(stat.style.format("{:,.4f}"), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(x=norm_series, nbins=45, title=f"Histogram · {norm_method}",
                                   labels={"x": norm_metric}, color_discrete_sequence=[COLORS["purple"]])
                show_chart(fig, 380)
            with c2:
                fig = px.box(y=norm_series, points=False, title=f"Boxplot · {norm_method}",
                             labels={"y": norm_metric}, color_discrete_sequence=[COLORS["cyan"]])
                show_chart(fig, 380)
            if {"Scale", "Z-Score"}.issubset(normalized):
                scale = normalized["Scale"]
                zscore = normalized["Z-Score"]
                if scale.shape == zscore.shape and scale.columns.equals(zscore.columns):
                    same = scale.equals(zscore)
                    st.info("Scale và Z-Score có cùng nội dung trong các file hiện tại." if same else
                            "Scale và Z-Score là hai file riêng; không giả định chúng trùng nhau.")

with tabs[5]:
    st.markdown("## Repository Explorer")
    st.markdown('<div class="panel-note">Tra cứu nhanh các repository trong mẫu theo bộ lọc đang áp dụng.</div>', unsafe_allow_html=True)
    sort_options = [c for c in ["star_count", "forks_count", "open_issues_count", "created_at", "age_years"] if c in filtered]
    sort_col = st.selectbox("Sắp xếp theo", sort_options, index=0,
                            format_func=lambda x: {"star_count":"Stars","forks_count":"Forks","open_issues_count":"Open issues","created_at":"Ngày tạo","age_years":"Tuổi repository"}[x])
    ascending = st.checkbox("Tăng dần", value=False)
    display_cols = [c for c in ["full_name", "primary_language_clean", "star_count", "forks_count",
                                "open_issues_count", "age_years", "topic_count", "has_issues", "has_wiki"] if c in filtered]
    table = filtered.sort_values(sort_col, ascending=ascending, na_position="last")[display_cols].head(500).copy()
    table = table.rename(columns={
        "full_name":"Repository", "primary_language_clean":"Language", "star_count":"Stars",
        "forks_count":"Forks", "open_issues_count":"Open issues", "age_years":"Age (years)",
        "topic_count":"Topics", "has_issues":"Issues enabled", "has_wiki":"Wiki enabled",
    })
    st.dataframe(table, use_container_width=True, hide_index=True, height=520)
    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download filtered CSV", data=csv_bytes, file_name="github_repositories_filtered.csv",
                       mime="text/csv", icon=":material/download:")

st.markdown(
    f'<div class="footer">{icon("github", "#7F91AD", 15)} &nbsp; GitHub Repository Popularity Analysis '
    f'· Data source: {raw_filename} · Snapshot: {snapshot}<br>'
    'Dashboard chỉ tạo feature trong bộ nhớ, không ghi đè hoặc biến đổi CSV gốc.</div>',
    unsafe_allow_html=True,
)