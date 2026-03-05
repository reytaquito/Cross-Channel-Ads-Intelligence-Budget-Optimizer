from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def optimize_budget(
    campaign_summary: pd.DataFrame,
    total_budget: float,
    max_scale: float = 2.0,
    steps: int = 600,
    min_share: float = 0.0,
) -> pd.DataFrame:
    if total_budget <= 0:
        raise ValueError("total_budget must be > 0")

    df = campaign_summary.copy()
    if df.empty:
        raise ValueError("campaign_summary is empty")

    historical_spend = df["spend"].clip(lower=0).to_numpy(dtype=float)
    historical_conv = df["conversions"].clip(lower=0).to_numpy(dtype=float)

    fallback_alpha = (historical_conv.sum() / np.sqrt(max(historical_spend.sum(), 1.0))) / max(len(df), 1)
    alphas = np.where(
        historical_spend > 0,
        historical_conv / np.sqrt(historical_spend + EPS),
        fallback_alpha,
    )
    alphas = np.clip(alphas, 1e-6, None)

    min_alloc = historical_spend * min_share
    max_alloc = np.where(
        historical_spend > 0,
        historical_spend * max_scale,
        total_budget / max(len(df), 1),
    )

    if min_alloc.sum() > total_budget:
        min_alloc *= total_budget / min_alloc.sum()

    allocation = min_alloc.copy()
    remaining = total_budget - allocation.sum()
    delta = max(total_budget / max(steps, 1), 0.01)

    safety_loops = max(steps * 25, 1000)
    for _ in range(safety_loops):
        if remaining <= EPS:
            break

        with np.errstate(divide="ignore", invalid="ignore"):
            marginals = alphas / (2.0 * np.sqrt(allocation + EPS))
        marginals[allocation >= max_alloc - EPS] = -np.inf

        best_idx = int(np.argmax(marginals))
        if not np.isfinite(marginals[best_idx]):
            break

        step = min(delta, remaining, max_alloc[best_idx] - allocation[best_idx])
        if step <= EPS:
            marginals[best_idx] = -np.inf
            continue

        allocation[best_idx] += step
        remaining -= step

    if remaining > EPS:
        room = np.clip(max_alloc - allocation, 0, None)
        room_sum = room.sum()
        if room_sum > EPS:
            allocation += remaining * (room / room_sum)
            remaining = 0

    projected_conversions = alphas * np.sqrt(allocation + EPS)

    out = df[
        ["platform", "campaign_id", "campaign_name", "spend", "conversions", "cpa", "efficiency_score"]
    ].copy()
    out = out.rename(
        columns={
            "spend": "current_spend",
            "conversions": "current_conversions",
            "cpa": "current_cpa",
        }
    )

    out["optimized_spend"] = allocation
    out["projected_conversions"] = projected_conversions
    out["projected_cpa"] = np.where(
        projected_conversions > EPS,
        allocation / projected_conversions,
        np.nan,
    )
    out["budget_change"] = out["optimized_spend"] - out["current_spend"]
    out["conversion_lift"] = out["projected_conversions"] - out["current_conversions"]

    return out.sort_values("projected_conversions", ascending=False).reset_index(drop=True)


def summarize_allocation(optimized_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        optimized_df.groupby("platform", as_index=False)[
            [
                "current_spend",
                "optimized_spend",
                "current_conversions",
                "projected_conversions",
            ]
        ]
        .sum(min_count=1)
        .fillna(0)
    )

    grouped["current_cpa"] = np.where(
        grouped["current_conversions"] > 0,
        grouped["current_spend"] / grouped["current_conversions"],
        np.nan,
    )
    grouped["projected_cpa"] = np.where(
        grouped["projected_conversions"] > 0,
        grouped["optimized_spend"] / grouped["projected_conversions"],
        np.nan,
    )
    grouped["budget_change"] = grouped["optimized_spend"] - grouped["current_spend"]
    grouped["conversion_lift"] = grouped["projected_conversions"] - grouped["current_conversions"]

    total_opt = grouped["optimized_spend"].sum()
    grouped["optimized_budget_share"] = np.where(
        total_opt > 0,
        grouped["optimized_spend"] / total_opt,
        np.nan,
    )

    return grouped.sort_values("projected_conversions", ascending=False).reset_index(drop=True)
