"""Spendwise FastAPI backend — overspend risk prediction."""

import json
import os
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spend_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_cols.json")

model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH, encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

app = FastAPI(title="Spendwise API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CategoryTotals(BaseModel):
    food: float = Field(0, ge=0)
    transport: float = Field(0, ge=0)
    misc: float = Field(0, ge=0)


class PredictRequest(BaseModel):
    monthly_budget: float = Field(..., gt=0)
    days_into_month: int = Field(..., ge=1, le=30)
    rent_or_pg_spent_so_far: float = Field(..., ge=0)
    daily_spending_log: List[float] = Field(..., min_length=1)
    category_totals: CategoryTotals = Field(default_factory=CategoryTotals)

    @model_validator(mode="after")
    def validate_log(self):
        if len(self.daily_spending_log) != self.days_into_month:
            raise ValueError(
                f"daily_spending_log must have exactly {self.days_into_month} "
                "entries (one per day elapsed)"
            )
        if any(v < 0 for v in self.daily_spending_log):
            raise ValueError("daily_spending_log amounts must be non-negative")
        return self


class PredictResponse(BaseModel):
    risk_tier: str
    confidence: float
    projected_month_end_spend: float
    suggested_daily_limit: float
    reasons: List[str]
    recommendation: str
    action_plan: List[str]


def compute_daily_stats(daily_log: List[float]) -> dict:
    arr = np.array(daily_log, dtype=float)
    avg_daily_spend = float(np.mean(arr))
    std_daily_spend = float(np.std(arr)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        trend_slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    else:
        trend_slope = 0.0
    max_daily_spend = float(np.max(arr))
    return {
        "avg_daily_spend": avg_daily_spend,
        "std_daily_spend": std_daily_spend,
        "trend_slope": trend_slope,
        "max_daily_spend": max_daily_spend,
    }


def project_month_end_spend(
    rent_or_pg: float, avg_daily_spend: float, trend_slope: float, days_into_month: int
) -> float:
    remaining_days = 30 - days_into_month
    projected_daily = avg_daily_spend + trend_slope * (remaining_days / 2)
    return rent_or_pg + projected_daily * remaining_days


def suggested_daily_limit(
    monthly_budget: float,
    rent_or_pg: float,
    daily_log: List[float],
    days_into_month: int,
) -> float:
    remaining_days = 30 - days_into_month
    if remaining_days <= 0:
        return 0.0
    daily_total = sum(daily_log)
    remaining_budget = monthly_budget - rent_or_pg - daily_total
    return remaining_budget / remaining_days


def category_shares(category_totals: CategoryTotals) -> dict[str, float]:
    totals = {
        "Food": category_totals.food,
        "Transport": category_totals.transport,
        "Misc": category_totals.misc,
    }
    grand = sum(totals.values())
    if grand <= 0:
        return {k: 0.0 for k in totals}
    return {k: v / grand * 100 for k, v in totals.items()}


def top_category(category_totals: CategoryTotals) -> tuple[str, float]:
    items = [
        ("Food", category_totals.food),
        ("Transport", category_totals.transport),
        ("Misc", category_totals.misc),
    ]
    return max(items, key=lambda x: x[1])


def daily_cut_to_target(
    monthly_budget: float,
    rent_or_pg: float,
    stats: dict,
    days_into_month: int,
    target_ratio: float = 0.90,
) -> float:
    remaining_days = 30 - days_into_month
    if remaining_days <= 0:
        return 0.0
    target = monthly_budget * target_ratio
    trend_adj = stats["trend_slope"] * (remaining_days / 2)
    required_avg = (target - rent_or_pg) / remaining_days
    cut = stats["avg_daily_spend"] + trend_adj - required_avg
    return max(0.0, cut)


def build_action_plan(
    req: PredictRequest,
    stats: dict,
    risk_tier: str,
    projected_month_end_spend: float,
    suggested_limit: float,
) -> List[str]:
    shares = category_shares(req.category_totals)
    top_name, top_amount = top_category(req.category_totals)
    remaining_days = 30 - req.days_into_month
    actions: List[str] = []

    if risk_tier == "SAFE":
        actions.append(
            "You are on a healthy pace — keep logging daily expenses so patterns stay visible."
        )
        if remaining_days > 0:
            actions.append(
                f"Stay at or below ₹{suggested_limit:,.0f}/day for the remaining "
                f"{remaining_days} days to finish the month under budget."
            )
        if top_amount > 0:
            actions.append(
                f"{top_name} is your largest daily category so far "
                f"({shares[top_name]:.0f}% of tracked spend) — keep it steady."
            )
        return actions[:3]

    cut_per_day = daily_cut_to_target(
        req.monthly_budget,
        req.rent_or_pg_spent_so_far,
        stats,
        req.days_into_month,
    )
    overshoot = projected_month_end_spend - req.monthly_budget * 0.90

    if top_amount > 0 and cut_per_day > 0:
        actions.append(
            f"Cut your {top_name} spend by about ₹{cut_per_day:,.0f}/day for the rest "
            f"of the month to get back on track "
            f"({top_name} is {shares[top_name]:.0f}% of your spending so far)."
        )
    elif cut_per_day > 0:
        actions.append(
            f"Reduce daily spending by about ₹{cut_per_day:,.0f}/day for the remaining "
            f"{remaining_days} days to bring your projection under 90% of budget."
        )

    if remaining_days > 0 and suggested_limit >= 0:
        actions.append(
            f"Overall daily cap: stay at or below ₹{max(0, suggested_limit):,.0f}/day "
            f"on food, transport, and misc combined."
        )

    if overshoot > 0:
        actions.append(
            f"You're projected ₹{overshoot:,.0f} over the safe threshold "
            f"(90% of ₹{req.monthly_budget:,.0f} budget) — act on the cuts above now."
        )

    return actions[:3]


def build_reasons(
    req: PredictRequest,
    stats: dict,
    risk_tier: str,
) -> List[str]:
    avg = stats["avg_daily_spend"]
    std = stats["std_daily_spend"]
    slope = stats["trend_slope"]
    max_day = stats["max_daily_spend"]
    daily_total = sum(req.daily_spending_log)
    remaining_days = 30 - req.days_into_month

    if risk_tier == "SAFE":
        reasons = ["Spending pace looks sustainable for the rest of the month."]
        if req.rent_or_pg_spent_so_far > 0:
            reasons.append(
                "Rent/PG is treated as a one-time payment already made — "
                "projections use your daily spending pattern, not a flat total."
            )
        if std > 0 and avg > 0 and std / avg < 0.25:
            reasons.append("Your daily spending has been relatively steady day to day.")
        return reasons

    reasons = []
    if req.rent_or_pg_spent_so_far > 0:
        total_so_far = req.rent_or_pg_spent_so_far + daily_total
        rent_share = req.rent_or_pg_spent_so_far / total_so_far * 100 if total_so_far > 0 else 0
        if rent_share >= 30:
            reasons.append(
                f"Your rent/PG payment (₹{req.rent_or_pg_spent_so_far:,.0f}) is already "
                "accounted for — projections focus on your daily spending pattern."
            )

    if avg > 0 and std / avg >= 0.35:
        reasons.append(
            f"Your spending has spikes — some days much higher than others "
            f"(avg ₹{avg:,.0f}/day, peak ₹{max_day:,.0f}/day)."
        )

    if slope > avg * 0.05 and req.days_into_month >= 3:
        reasons.append(
            "Your daily spending has been increasing over the past several days — "
            "the trend is factored into the month-end projection."
        )
    elif slope < -avg * 0.05 and req.days_into_month >= 3:
        reasons.append(
            "Your daily spending has been slowing down recently — "
            "that deceleration improves your outlook vs a flat average."
        )
    else:
        reasons.append(
            f"Daily spending averages ₹{avg:,.0f}/day over {req.days_into_month} days "
            f"(₹{daily_total:,.0f} total so far)."
        )

    projected_daily = avg + slope * (remaining_days / 2)
    reasons.append(
        f"Trend-adjusted daily pace for remaining {remaining_days} days: "
        f"₹{projected_daily:,.0f}/day projected."
    )

    if risk_tier == "HIGH_RISK":
        reasons.append(
            "Pattern signals (trend + volatility) point to overspend even if today's total looks manageable."
        )
    elif risk_tier == "CAUTION":
        reasons.append(
            "Your spending pattern suggests daily costs will add up over the rest of the month."
        )
    return reasons


def recommendation_for(risk_tier: str, suggested_limit: float) -> str:
    if risk_tier == "SAFE":
        return "Keep tracking daily spends. You are on a healthy pace for month-end."
    if risk_tier == "CAUTION":
        return (
            f"Cap daily spending at ₹{suggested_limit:,.0f} for the rest of the month "
            "and watch for upward trends or spike days."
        )
    return (
        f"Act now: aim for ₹{max(0, suggested_limit):,.0f}/day or less — "
        "cut discretionary spend and avoid spike days until month-end."
    )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Spendwise API"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    stats = compute_daily_stats(req.daily_spending_log)
    projected_month_end_spend = project_month_end_spend(
        req.rent_or_pg_spent_so_far,
        stats["avg_daily_spend"],
        stats["trend_slope"],
        req.days_into_month,
    )
    limit = suggested_daily_limit(
        req.monthly_budget,
        req.rent_or_pg_spent_so_far,
        req.daily_spending_log,
        req.days_into_month,
    )

    features = {
        "monthly_budget": req.monthly_budget,
        "days_into_month": req.days_into_month,
        "rent_or_pg_spent_so_far": req.rent_or_pg_spent_so_far,
        **stats,
    }
    X = np.array([[features[col] for col in FEATURE_COLS]])

    risk_tier = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(max(proba))

    return PredictResponse(
        risk_tier=risk_tier,
        confidence=round(confidence, 3),
        projected_month_end_spend=round(projected_month_end_spend, 2),
        suggested_daily_limit=round(limit, 2),
        reasons=build_reasons(req, stats, risk_tier),
        recommendation=recommendation_for(risk_tier, limit),
        action_plan=build_action_plan(
            req, stats, risk_tier, projected_month_end_spend, limit
        ),
    )
