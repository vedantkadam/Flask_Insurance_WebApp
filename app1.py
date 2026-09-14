"""
app1.py
----------------------------------------------------
Flask backend for the InsuranceIQ analytics + premium
prediction dashboard.

Works seamlessly on both Local Machine and Cloud Hosts
(Render / Railway / Heroku).
----------------------------------------------------
"""

from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# --------------------------------------------------
# Safe Absolute Path Resolution
# --------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "model_metrics.pkl")
CSV_FILE = os.path.join(
    BASE_DIR, "insurance_linear_regression_raw_2000.csv"
)

NUMERIC_FEATURES = [
    "Customer_Age",
    "Annual_Income",
    "Claim_History_Count",
    "Previous_Claim_Amount",
    "Policy_Tenure_Years",
    "Risk_Score",
    "Insured_Value",
]
CATEGORICAL_FEATURES = ["Policy_Type", "Customer_Segment"]
TARGET = "Annual_Insurance_Premium"


# --------------------------------------------------
# Load Model & Metrics Safely
# --------------------------------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} not found. Place your trained model.pkl at: {MODEL_FILE}"
    )

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

model_metrics = {}
if os.path.exists(METRICS_FILE):
    try:
        with open(METRICS_FILE, "rb") as file:
            model_metrics = pickle.load(file)
            print(f"[SUCCESS] Loaded model_metrics.pkl: {model_metrics}")
    except Exception as e:
        print(f"[ERROR] Could not load model_metrics.pkl: {e}")
        model_metrics = {}
else:
    print(
        f"[INFO] {METRICS_FILE} not found. Proceeding with empty metrics."
    )


# --------------------------------------------------
# Data Cleaning (Analytics Dashboard)
# --------------------------------------------------

POLICY_TYPE_MAP = {
    "auto": "Auto",
    "home": "Home",
    "health": "Health",
    "life": "Life",
    "lifee": "Life",
    "travel": "Travel",
}

SEGMENT_MAP = {
    "mass": "Mass",
    "affluent": "Affluent",
    "premium": "Premium",
    "corporate": "Corporate",
    "corporrate": "Corporate",
}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Policy_Type"] = (
        df["Policy_Type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(POLICY_TYPE_MAP)
    )
    df["Customer_Segment"] = (
        df["Customer_Segment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SEGMENT_MAP)
    )

    df["Customer_Age"] = df["Customer_Age"].clip(lower=18, upper=100)
    df["Annual_Income"] = df["Annual_Income"].clip(lower=0)
    df["Previous_Claim_Amount"] = df["Previous_Claim_Amount"].clip(lower=0)
    df["Insured_Value"] = df["Insured_Value"].clip(lower=0)
    df["Claim_History_Count"] = df["Claim_History_Count"].clip(lower=0)
    df["Policy_Tenure_Years"] = df["Policy_Tenure_Years"].clip(lower=0)
    df["Risk_Score"] = df["Risk_Score"].clip(lower=0, upper=100)

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].clip(lower=0)

    for col in ["Annual_Income", "Previous_Claim_Amount", "Insured_Value"]:
        low, high = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=low, upper=high)

    return df


if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found at: {CSV_FILE}")

dataset = clean_dataframe(pd.read_csv(CSV_FILE))


# --------------------------------------------------
# Home Page Route
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# --------------------------------------------------
# Analytics API Endpoint
# --------------------------------------------------

def _safe_round(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), digits)


@app.route("/api/analytics")
def analytics():
    df = dataset

    kpis = {
        "total_policies": int(len(df)),
        "average_premium": _safe_round(df[TARGET].mean()),
        "average_risk_score": _safe_round(df["Risk_Score"].mean()),
        "average_insured_value": _safe_round(df["Insured_Value"].mean()),
    }

    # Premium distribution (histogram buckets)
    premiums = df[TARGET].dropna()
    counts, bin_edges = np.histogram(premiums, bins=10)
    premium_distribution = {
        "labels": [
            f"{int(bin_edges[i]/1000)}k-{int(bin_edges[i+1]/1000)}k"
            for i in range(len(bin_edges) - 1)
        ],
        "counts": counts.tolist(),
    }

    # Average premium by policy type
    by_policy = df.groupby("Policy_Type")[TARGET].mean().sort_values(ascending=False)
    premium_by_policy_type = {
        "labels": by_policy.index.tolist(),
        "values": [_safe_round(v) for v in by_policy.values],
    }

    # Risk score vs premium (scatter, sampled)
    scatter_df = df[["Risk_Score", TARGET]].dropna()
    if len(scatter_df) > 300:
        scatter_df = scatter_df.sample(300, random_state=42)
    risk_vs_premium = [
        {"x": _safe_round(r), "y": _safe_round(p)}
        for r, p in zip(scatter_df["Risk_Score"], scatter_df[TARGET])
    ]

    # Policies by customer segment
    by_segment = df["Customer_Segment"].value_counts()
    segment_analysis = {
        "labels": by_segment.index.tolist(),
        "values": by_segment.values.tolist(),
    }

    # Insured value vs premium (scatter, sampled)
    scatter_df2 = df[["Insured_Value", TARGET]].dropna()
    if len(scatter_df2) > 300:
        scatter_df2 = scatter_df2.sample(300, random_state=42)
    value_vs_premium = [
        {"x": _safe_round(v), "y": _safe_round(p)}
        for v, p in zip(scatter_df2["Insured_Value"], scatter_df2[TARGET])
    ]

    # Data-driven insights
    top_policy_premium = by_policy.index[0]
    by_segment_risk = df.groupby("Customer_Segment")["Risk_Score"].mean().sort_values(
        ascending=False
    )
    top_segment_risk = by_segment_risk.index[0]
    risk_premium_corr = df["Risk_Score"].corr(df[TARGET])
    pct_with_claims = (df["Claim_History_Count"] > 0).mean() * 100
    by_policy_value = (
        df.groupby("Policy_Type")["Insured_Value"].mean().sort_values(ascending=False)
    )
    top_policy_value = by_policy_value.index[0]

    if risk_premium_corr > 0.3:
        corr_strength = "a moderate positive relationship"
    elif risk_premium_corr > 0.1:
        corr_strength = "a weak positive relationship"
    elif risk_premium_corr < -0.1:
        corr_strength = "an inverse relationship"
    else:
        corr_strength = "little to no linear relationship"

    insights = [
        {
            "title": "Highest-premium policy type",
            "text": f"{top_policy_premium} policies carry the highest average annual "
                    f"premium at \u20b9{by_policy.iloc[0]:,.0f}.",
        },
        {
            "title": "Highest-risk segment",
            "text": f"{top_segment_risk} customers have the highest average risk score "
                    f"at {by_segment_risk.iloc[0]:.1f}.",
        },
        {
            "title": "Risk and premium",
            "text": f"Risk score and annual premium show {corr_strength} "
                    f"(correlation coefficient of {risk_premium_corr:.2f}).",
        },
        {
            "title": "Claim history",
            "text": f"{pct_with_claims:.1f}% of customers have at least one "
                    f"prior claim on record.",
        },
        {
            "title": "Highest insured value",
            "text": f"{top_policy_value} policies have the highest average insured "
                    f"value at \u20b9{by_policy_value.iloc[0]:,.0f}.",
        },
    ]

    dropdown_options = {
        "policy_types": sorted(df["Policy_Type"].dropna().unique().tolist()),
        "customer_segments": sorted(df["Customer_Segment"].dropna().unique().tolist()),
    }

    return jsonify({
        "kpis": kpis,
        "premium_distribution": premium_distribution,
        "premium_by_policy_type": premium_by_policy_type,
        "risk_vs_premium": risk_vs_premium,
        "segment_analysis": segment_analysis,
        "value_vs_premium": value_vs_premium,
        "insights": insights,
        "dropdown_options": dropdown_options,
        "model_metrics": model_metrics,
    })


# --------------------------------------------------
# Prediction API Endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        numeric_values = {}
        for field in NUMERIC_FEATURES:
            if field not in data:
                raise KeyError(field)
            numeric_values[field] = float(data[field])

        categorical_values = {}
        for field in CATEGORICAL_FEATURES:
            if field not in data or not str(data[field]).strip():
                raise KeyError(field)
            categorical_values[field] = str(data[field]).strip()

        row = {**numeric_values, **categorical_values}
        input_data = pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        prediction = model.predict(input_data)[0]
        prediction = max(0.0, float(prediction))

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2)
        })

    except KeyError as e:
        return jsonify({
            "success": False,
            "error": f"Missing input: {str(e)}"
        }), 400

    except ValueError:
        return jsonify({
            "success": False,
            "error": "Please enter valid numbers."
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# Adaptive Server Execution (Local vs Cloud)
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    is_cloud = "PORT" in os.environ or "RENDER" in os.environ

    app.run(
        host="0.0.0.0" if is_cloud else "127.0.0.1",
        port=port,
        debug=not is_cloud
    )