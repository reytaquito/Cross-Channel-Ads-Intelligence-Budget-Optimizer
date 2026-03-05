from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_pipeline import (
    daily_trends,
    generate_alerts,
    load_all_platforms,
    summarize_campaigns,
    summarize_platform,
)
from src.optimizer import optimize_budget, summarize_allocation

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PATHS = {
    "facebook": str(BASE_DIR / "Datasets" / "01_facebook_ads.csv"),
    "google": str(BASE_DIR / "Datasets" / "02_google_ads.csv"),
    "tiktok": str(BASE_DIR / "Datasets" / "03_tiktok_ads.csv"),
}


@st.cache_data(show_spinner=False)
def load_data(facebook_source, google_source, tiktok_source) -> pd.DataFrame:
    return load_all_platforms(
        {
            "facebook": facebook_source,
            "google": google_source,
            "tiktok": tiktok_source,
        }
    )


def _currency(v: float) -> str:
    return f"${v:,.2f}"


def _pct(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:.2%}"


def main() -> None:
    st.set_page_config(page_title="Cross-Channel Ads Intelligence", layout="wide")
    st.title("Cross-Channel Ads Intelligence + Budget Optimizer")
    st.caption("Unifies Facebook, Google, and TikTok performance into one decision dashboard.")

    st.sidebar.header("Data Sources")
    fb_file = st.sidebar.file_uploader("Facebook CSV", type="csv")
    gg_file = st.sidebar.file_uploader("Google CSV", type="csv")
    tt_file = st.sidebar.file_uploader("TikTok CSV", type="csv")

    fb_path = st.sidebar.text_input("Facebook path", value=DEFAULT_PATHS["facebook"])
    gg_path = st.sidebar.text_input("Google path", value=DEFAULT_PATHS["google"])
    tt_path = st.sidebar.text_input("TikTok path", value=DEFAULT_PATHS["tiktok"])

    facebook_source = fb_file if fb_file is not None else fb_path
    google_source = gg_file if gg_file is not None else gg_path
    tiktok_source = tt_file if tt_file is not None else tt_path

    try:
        df = load_data(facebook_source, google_source, tiktok_source)
    except Exception as exc:
        st.error(f"Could not load CSVs: {exc}")
        st.stop()

    if df.empty:
        st.warning("No rows were loaded.")
        st.stop()

    platform_summary = summarize_platform(df)
    campaign_summary = summarize_campaigns(df)
    trends = daily_trends(df)

    total_spend = float(df["spend"].sum())
    total_conv = float(df["conversions"].sum())
    total_clicks = float(df["clicks"].sum())
    total_impressions = float(df["impressions"].sum())
    overall_ctr = total_clicks / total_impressions if total_impressions else float("nan")
    overall_cpa = total_spend / total_conv if total_conv else float("nan")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", _currency(total_spend))
    c2.metric("Total Conversions", f"{total_conv:,.0f}")
    c3.metric("Overall CTR", _pct(overall_ctr))
    c4.metric("Overall CPA", _currency(overall_cpa) if pd.notna(overall_cpa) else "-")

    tab_overview, tab_campaigns, tab_optimizer, tab_alerts = st.tabs(
        ["Overview", "Campaign Deep Dive", "Budget Optimizer", "Alerts"]
    )

    with tab_overview:
        left, right = st.columns((1.2, 1))

        with left:
            fig_spend = px.line(
                trends,
                x="date",
                y="spend",
                color="platform",
                markers=True,
                title="Daily Spend by Platform",
            )
            st.plotly_chart(fig_spend, use_container_width=True)

            fig_conv = px.line(
                trends,
                x="date",
                y="conversions",
                color="platform",
                markers=True,
                title="Daily Conversions by Platform",
            )
            st.plotly_chart(fig_conv, use_container_width=True)

        with right:
            platform_table = platform_summary.copy()
            platform_table["ctr"] = platform_table["ctr"].map(lambda v: f"{v:.2%}")
            platform_table["cpc"] = platform_table["cpc"].map(lambda v: f"${v:.2f}")
            platform_table["cpa"] = platform_table["cpa"].map(lambda v: f"${v:.2f}")
            platform_table["roas"] = platform_table["roas"].map(lambda v: "-" if pd.isna(v) else f"{v:.2f}")
            st.subheader("Platform Performance")
            st.dataframe(
                platform_table[
                    ["platform", "spend", "conversions", "ctr", "cpc", "cpa", "roas"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            fig_cpa = px.bar(
                platform_summary,
                x="platform",
                y="cpa",
                color="platform",
                title="CPA by Platform",
            )
            st.plotly_chart(fig_cpa, use_container_width=True)

    with tab_campaigns:
        st.subheader("Campaign Efficiency Map")
        fig_scatter = px.scatter(
            campaign_summary,
            x="spend",
            y="conversions",
            color="platform",
            size="clicks",
            hover_data=["campaign_name", "ctr", "cvr", "cpa", "roas", "efficiency_score"],
            title="Spend vs Conversions (bubble size = clicks)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        c_top, c_bottom = st.columns(2)
        with c_top:
            st.markdown("**Top 5 Efficient Campaigns**")
            st.dataframe(
                campaign_summary.head(5)[
                    ["platform", "campaign_name", "spend", "conversions", "ctr", "cvr", "cpa", "efficiency_score"]
                ],
                use_container_width=True,
                hide_index=True,
            )
        with c_bottom:
            st.markdown("**Bottom 5 Efficient Campaigns**")
            st.dataframe(
                campaign_summary.tail(5)[
                    ["platform", "campaign_name", "spend", "conversions", "ctr", "cvr", "cpa", "efficiency_score"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_optimizer:
        st.subheader("Budget Reallocation Simulator")
        budget = st.number_input(
            "Total budget to allocate",
            min_value=100.0,
            value=float(round(total_spend, 2)),
            step=50.0,
        )
        max_scale = st.slider("Max spend scale per campaign", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

        optimized = optimize_budget(campaign_summary, budget, max_scale=max_scale)
        platform_alloc = summarize_allocation(optimized)

        current_spend = float(optimized["current_spend"].sum())
        current_conv = float(optimized["current_conversions"].sum())
        projected_conv = float(optimized["projected_conversions"].sum())
        current_cpa = current_spend / current_conv if current_conv else float("nan")
        projected_cpa = budget / projected_conv if projected_conv else float("nan")

        o1, o2, o3 = st.columns(3)
        o1.metric("Projected Conversions", f"{projected_conv:,.1f}", delta=f"{projected_conv - current_conv:+.1f}")
        projected_cpa_label = _currency(projected_cpa) if pd.notna(projected_cpa) else "-"
        cpa_delta = f"{(projected_cpa - current_cpa):+.2f}" if pd.notna(projected_cpa) and pd.notna(current_cpa) else None
        o2.metric("Projected CPA", projected_cpa_label, delta=cpa_delta)
        o3.metric("Budget", _currency(budget))

        melted = optimized.melt(
            id_vars=["platform", "campaign_name"],
            value_vars=["current_spend", "optimized_spend"],
            var_name="scenario",
            value_name="spend",
        )
        fig_alloc = px.bar(
            melted,
            x="campaign_name",
            y="spend",
            color="scenario",
            facet_row="platform",
            barmode="group",
            title="Current vs Optimized Campaign Budget",
            height=800,
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

        st.markdown("**Platform Reallocation Summary**")
        st.dataframe(platform_alloc, use_container_width=True, hide_index=True)

        st.markdown("**Campaign-Level Recommendation**")
        st.dataframe(
            optimized[
                [
                    "platform",
                    "campaign_name",
                    "current_spend",
                    "optimized_spend",
                    "budget_change",
                    "current_conversions",
                    "projected_conversions",
                    "conversion_lift",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_alerts:
        st.subheader("Automated Performance Alerts")
        alerts = generate_alerts(platform_summary, campaign_summary)
        for alert in alerts:
            st.write(f"- {alert}")

        st.subheader("Raw Unified Data")
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
