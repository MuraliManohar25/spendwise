"""
Generate synthetic student monthly spending scenarios for Spendwise.

This dataset is synthetic and rule-informed: there is no public dataset that
matches this specific student-budgeting-trajectory problem (budget + partial
month daily spend pattern -> overspend risk). Labels are derived from a
trend-aware projected month-end spend vs budget; the RandomForest model
learns from derived pattern features rather than hardcoding rules at
prediction time.
"""

import os

import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 3000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "spending_dataset.csv")


def label_risk(projected_spend: float, monthly_budget: float) -> str:
    ratio = projected_spend / monthly_budget
    if ratio < 0.90:
        return "SAFE"
    if ratio <= 1.15:
        return "CAUTION"
    return "HIGH_RISK"


def compute_daily_stats(daily_log: list[float]) -> tuple[float, float, float, float]:
    arr = np.array(daily_log, dtype=float)
    avg_daily_spend = float(np.mean(arr))
    std_daily_spend = float(np.std(arr)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        trend_slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    else:
        trend_slope = 0.0
    max_daily_spend = float(np.max(arr))
    return avg_daily_spend, std_daily_spend, trend_slope, max_daily_spend


def project_month_end_spend(
    rent_or_pg: float, avg_daily_spend: float, trend_slope: float, days_into_month: int
) -> float:
    # Trend and volatility matter more than a flat total: two students with
    # the same spend-so-far can have very different month-end outcomes if one
    # is accelerating or spiky while the other is steady.
    remaining_days = 30 - days_into_month
    projected_daily = avg_daily_spend + trend_slope * (remaining_days / 2)
    return rent_or_pg + projected_daily * remaining_days


def simulate_daily_log(
    days_into_month: int,
    base_daily: float,
    pattern: str,
) -> list[float]:
    daily_log = []
    for day in range(days_into_month):
        if pattern == "steady":
            amount = base_daily * np.random.uniform(0.88, 1.12)
        elif pattern == "spiky":
            amount = base_daily * np.random.uniform(0.45, 2.1)
            if day % 7 in (5, 6):
                amount *= np.random.uniform(1.25, 1.85)
        elif pattern == "upward":
            progress = day / max(days_into_month - 1, 1)
            amount = base_daily * (0.65 + 0.70 * progress) * np.random.uniform(0.9, 1.1)
        else:  # downward
            progress = day / max(days_into_month - 1, 1)
            amount = base_daily * (1.35 - 0.70 * progress) * np.random.uniform(0.9, 1.1)
        daily_log.append(max(50.0, amount))
    return daily_log


rows = []
for _ in range(N_ROWS):
    monthly_budget = float(np.random.randint(8000, 25001))
    days_into_month = int(np.random.randint(1, 31))

    rent_share = np.random.uniform(0.35, 0.50)
    rent_or_pg_spent_so_far = monthly_budget * rent_share * np.random.uniform(0.92, 1.08)

    pace_factor = np.random.uniform(0.65, 1.45)
    daily_pool = monthly_budget - rent_or_pg_spent_so_far
    base_daily = (daily_pool / 30) * pace_factor

    pattern = np.random.choice(["steady", "spiky", "upward", "downward"], p=[0.30, 0.25, 0.25, 0.20])
    daily_log = simulate_daily_log(days_into_month, base_daily, pattern)

    avg_daily_spend, std_daily_spend, trend_slope, max_daily_spend = compute_daily_stats(daily_log)
    projected_spend = project_month_end_spend(
        rent_or_pg_spent_so_far, avg_daily_spend, trend_slope, days_into_month
    )
    overspend_risk = label_risk(projected_spend, monthly_budget)

    rows.append(
        {
            "monthly_budget": round(monthly_budget, 2),
            "days_into_month": days_into_month,
            "rent_or_pg_spent_so_far": round(rent_or_pg_spent_so_far, 2),
            "avg_daily_spend": round(avg_daily_spend, 2),
            "std_daily_spend": round(std_daily_spend, 2),
            "trend_slope": round(trend_slope, 4),
            "max_daily_spend": round(max_daily_spend, 2),
            "overspend_risk": overspend_risk,
        }
    )

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
print("\nLabel distribution:")
print(df["overspend_risk"].value_counts())
print("\nLabel distribution (%):")
print(df["overspend_risk"].value_counts(normalize=True).mul(100).round(1))
