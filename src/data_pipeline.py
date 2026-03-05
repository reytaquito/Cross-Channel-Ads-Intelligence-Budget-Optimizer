from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

CsvSource = Union[str, Path, Any]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _read_csv(source: CsvSource) -> pd.DataFrame:
    if hasattr(source, "read"):
        return pd.read_csv(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _standardize_facebook(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["platform"] = "facebook"
    out["entity_id"] = out.get("ad_set_id")
    out["entity_name"] = out.get("ad_set_name")
    out["spend"] = out.get("spend")
    out["conversion_value"] = np.nan
    out["video_watch_25"] = np.nan
    out["video_watch_50"] = np.nan
    out["video_watch_75"] = np.nan
    out["video_watch_100"] = np.nan
    out["likes"] = np.nan
    out["shares"] = np.nan
    out["comments"] = np.nan
    out["quality_score"] = np.nan
    out["search_impression_share"] = np.nan
    return out


def _standardize_google(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["platform"] = "google"
    out["entity_id"] = out.get("ad_group_id")
    out["entity_name"] = out.get("ad_group_name")
    out["spend"] = out.get("cost")
    out["video_views"] = np.nan
    out["video_watch_25"] = np.nan
    out["video_watch_50"] = np.nan
    out["video_watch_75"] = np.nan
    out["video_watch_100"] = np.nan
    out["likes"] = np.nan
    out["shares"] = np.nan
    out["comments"] = np.nan
    out["engagement_rate"] = np.nan
    out["reach"] = np.nan
    out["frequency"] = np.nan
    return out


def _standardize_tiktok(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["platform"] = "tiktok"
    out["entity_id"] = out.get("adgroup_id")
    out["entity_name"] = out.get("adgroup_name")
    out["spend"] = out.get("cost")
    out["conversion_value"] = np.nan
    out["reach"] = np.nan
    out["frequency"] = np.nan
    out["quality_score"] = np.nan
    out["search_impression_share"] = np.nan

    interactions = out[["likes", "shares", "comments"]].sum(axis=1)
    out["engagement_rate"] = _safe_div(interactions, out["impressions"])
    return out


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "impressions",
        "clicks",
        "spend",
        "conversions",
        "conversion_value",
        "video_views",
        "video_watch_25",
        "video_watch_50",
        "video_watch_75",
        "video_watch_100",
        "likes",
        "shares",
        "comments",
        "engagement_rate",
        "reach",
        "frequency",
        "quality_score",
        "search_impression_share",
    ]

    _to_numeric(df, numeric_cols)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["ctr"] = _safe_div(df["clicks"], df["impressions"])
    df["cpc"] = _safe_div(df["spend"], df["clicks"])
    df["cpm"] = _safe_div(1000 * df["spend"], df["impressions"])
    df["cvr"] = _safe_div(df["conversions"], df["clicks"])
    df["cpa"] = _safe_div(df["spend"], df["conversions"])
    df["roas"] = _safe_div(df["conversion_value"], df["spend"])
    df["video_completion_rate"] = _safe_div(df["video_watch_100"], df["video_views"])

    keep_cols = [
        "platform",
        "date",
        "campaign_id",
        "campaign_name",
        "entity_id",
        "entity_name",
        "impressions",
        "clicks",
        "spend",
        "conversions",
        "conversion_value",
        "video_views",
        "video_watch_25",
        "video_watch_50",
        "video_watch_75",
        "video_watch_100",
        "likes",
        "shares",
        "comments",
        "engagement_rate",
        "reach",
        "frequency",
        "quality_score",
        "search_impression_share",
        "ctr",
        "cpc",
        "cpm",
        "cvr",
        "cpa",
        "roas",
        "video_completion_rate",
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[keep_cols].sort_values(["date", "platform", "campaign_name"]).reset_index(drop=True)


def load_all_platforms(sources: Dict[str, CsvSource]) -> pd.DataFrame:
    facebook = _finalize(_standardize_facebook(_read_csv(sources["facebook"])))
    google = _finalize(_standardize_google(_read_csv(sources["google"])))
    tiktok = _finalize(_standardize_tiktok(_read_csv(sources["tiktok"])))
    return pd.concat([facebook, google, tiktok], ignore_index=True)


def _add_aggregate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "conversion_value" not in out.columns:
        out["conversion_value"] = np.nan

    out["ctr"] = _safe_div(out["clicks"], out["impressions"])
    out["cpc"] = _safe_div(out["spend"], out["clicks"])
    out["cpm"] = _safe_div(1000 * out["spend"], out["impressions"])
    out["cvr"] = _safe_div(out["conversions"], out["clicks"])
    out["cpa"] = _safe_div(out["spend"], out["conversions"])
    out["roas"] = _safe_div(out["conversion_value"], out["spend"])
    return out


def summarize_platform(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("platform", as_index=False)[
            [
                "impressions",
                "clicks",
                "spend",
                "conversions",
                "conversion_value",
                "video_views",
                "likes",
                "shares",
                "comments",
            ]
        ]
        .sum(min_count=1)
        .fillna(0)
    )
    return _add_aggregate_kpis(grouped).sort_values("conversions", ascending=False)


def _zscore(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / (series.std(ddof=0) + 1e-9)


def summarize_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["platform", "campaign_id", "campaign_name"], as_index=False)[
            [
                "impressions",
                "clicks",
                "spend",
                "conversions",
                "conversion_value",
                "video_views",
                "video_watch_100",
                "likes",
                "shares",
                "comments",
            ]
        ]
        .sum(min_count=1)
        .fillna(0)
    )
    grouped = _add_aggregate_kpis(grouped)
    grouped["video_completion_rate"] = _safe_div(grouped["video_watch_100"], grouped["video_views"])

    roas_filled = grouped["roas"].fillna(0)
    grouped["efficiency_score"] = (
        _zscore(grouped["ctr"].fillna(0))
        + _zscore(grouped["cvr"].fillna(0))
        + 0.5 * _zscore(roas_filled)
        - _zscore(grouped["cpa"].replace(np.inf, np.nan).fillna(grouped["cpa"].median()))
        - 0.5 * _zscore(grouped["cpc"].replace(np.inf, np.nan).fillna(grouped["cpc"].median()))
    )
    return grouped.sort_values("efficiency_score", ascending=False).reset_index(drop=True)


def daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["date", "platform"], as_index=False)[
            ["impressions", "clicks", "spend", "conversions", "conversion_value", "video_views"]
        ]
        .sum(min_count=1)
        .fillna(0)
    )
    return _add_aggregate_kpis(grouped)


def generate_alerts(platform_summary: pd.DataFrame, campaign_summary: pd.DataFrame) -> list[str]:
    alerts: list[str] = []

    median_cpa = campaign_summary["cpa"].replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(median_cpa):
        high_cpa = campaign_summary[campaign_summary["cpa"] > median_cpa * 1.35]
        for _, row in high_cpa.head(4).iterrows():
            alerts.append(
                f"High CPA: {row['platform']} / {row['campaign_name']} at ${row['cpa']:.2f} (median ${median_cpa:.2f})."
            )

    median_ctr = campaign_summary["ctr"].replace([np.inf, -np.inf], np.nan).median()
    if pd.notna(median_ctr):
        low_ctr = campaign_summary[campaign_summary["ctr"] < median_ctr * 0.7]
        for _, row in low_ctr.head(4).iterrows():
            alerts.append(
                f"Low CTR: {row['platform']} / {row['campaign_name']} at {row['ctr']:.2%} (median {median_ctr:.2%})."
            )

    roas_rows = campaign_summary[campaign_summary["roas"].notna()]
    if not roas_rows.empty:
        weak_roas = roas_rows[roas_rows["roas"] < 1.2]
        for _, row in weak_roas.head(4).iterrows():
            alerts.append(
                f"ROAS below target: {row['platform']} / {row['campaign_name']} at {row['roas']:.2f}."
            )

    if not alerts:
        alerts.append("No critical alerts detected with current thresholds.")

    return alerts
