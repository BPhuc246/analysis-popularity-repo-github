"""Shared data-loading and feature helpers for the GitHub repository analysis.

No source CSV is modified. Derived columns are created only in an in-memory copy.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FILENAME = "01_github_repos_chua_chuan_hoa_raw.csv"
NORMALIZED_FILENAMES = {
    "Scale": "scale_github_repos_binned.csv",
    "Z-Score": "zscore_github_repos_binned.csv",
    "Min-Max": "minmax_github_repos_binned.csv",
}
NUMERIC_METRICS = ["star_count", "forks_count", "open_issues_count", "size_kb"]
POPULARITY_METRICS = ["star_count", "forks_count"]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve the directory containing the raw and optional normalized CSVs."""
    if data_dir is not None:
        candidate = Path(data_dir).expanduser().resolve()
        if not (candidate / RAW_FILENAME).is_file():
            raise FileNotFoundError(f"Không tìm thấy {RAW_FILENAME} trong {candidate}")
        return candidate

    env_dir = os.getenv("GITHUB_REPO_DATA_DIR")
    if env_dir:
        candidate = Path(env_dir).expanduser().resolve()
        if (candidate / RAW_FILENAME).is_file():
            return candidate
        raise FileNotFoundError(
            f"GITHUB_REPO_DATA_DIR={candidate} không chứa {RAW_FILENAME}."
        )

    candidates = [
        PROJECT_ROOT / "data" / "input",
        PROJECT_ROOT / "data",
        PROJECT_ROOT,
        PROJECT_ROOT / "data" / "03",
        PROJECT_ROOT / "data" / "01",
        Path.cwd(),
        Path.cwd().parent,
    ]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / RAW_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy CSV raw. Đặt các CSV vào data/input/ hoặc cấu hình "
        "biến môi trường GITHUB_REPO_DATA_DIR."
    )


def load_raw_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load the supplied raw CSV without in-place or on-disk modification."""
    path = resolve_data_dir(data_dir) / RAW_FILENAME
    return pd.read_csv(path, low_memory=False)


def load_normalized_frames(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load any available normalized files by their supplied filenames."""
    directory = resolve_data_dir(data_dir)
    frames: dict[str, pd.DataFrame] = {}
    for method, filename in NORMALIZED_FILENAMES.items():
        path = directory / filename
        if path.is_file():
            frames[method] = pd.read_csv(path, low_memory=False)
    return frames


def parse_json_list(value: Any) -> list:
    """Parse JSON-encoded array cells; invalid/non-array values become []."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []
    return parsed if isinstance(parsed, list) else []


def prepare_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Create analysis-only parsed/date features on a new DataFrame copy."""
    df = raw.copy()
    for col in ["created_at", "updated_at", "pushed_at", "collected_at"]:
        if col in df.columns:
            df[f"{col}_dt"] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if {"created_at_dt", "collected_at_dt"}.issubset(df.columns):
        delta = df["collected_at_dt"] - df["created_at_dt"]
        df["age_days"] = delta.dt.total_seconds() / 86400.0
        df["age_years"] = df["age_days"] / 365.25
        df["created_year"] = df["created_at_dt"].dt.year.astype("Int64")

    if "topics" in df.columns:
        df["topics_list"] = df["topics"].map(parse_json_list)
        df["topic_count"] = df["topics_list"].map(
            lambda values: len({str(v).strip().casefold() for v in values if str(v).strip()})
        )
    else:
        df["topics_list"] = [[] for _ in range(len(df))]
        df["topic_count"] = 0

    if "languages" in df.columns:
        df["languages_list"] = df["languages"].map(parse_json_list)
    else:
        df["languages_list"] = [[] for _ in range(len(df))]

    if "primary_language" in df.columns:
        df["primary_language_clean"] = df["primary_language"].astype("string").str.strip()
        df["primary_language_key"] = df["primary_language_clean"].str.casefold()
    return df


def get_programming_language_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude blank labels and the literal 'text' from programming-language analysis."""
    key_col = "primary_language_key"
    if key_col not in df:
        out = df.copy()
        out[key_col] = out["primary_language"].astype("string").str.strip().str.casefold()
    else:
        out = df.copy()
    mask = out[key_col].notna() & out[key_col].ne("") & out[key_col].ne("text")
    return out.loc[mask].copy()


def topic_frequency(df: pd.DataFrame) -> pd.Series:
    """Count repository incidence for each unique topic, not raw repeated tokens."""
    counts: Counter[str] = Counter()
    for values in df.get("topics_list", pd.Series([[]] * len(df))):
        unique_values = {str(v).strip().casefold() for v in values if str(v).strip()}
        counts.update(unique_values)
    return pd.Series(dict(counts), dtype="int64").sort_values(ascending=False)


def save_figure(fig, path: str | Path, dpi: int = 300) -> Path:
    """Save a matplotlib/seaborn figure to a reproducible output path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target
