
# Ethical AI — Bias & Fairness Audit for Traffic Prediction Model
#
# PURPOSE:
#   A model that is "accurate on average" can still be systematically
#   worse for certain groups or conditions. This is called algorithmic bias.
#
#   For a traffic management system used by city councils, bias means:
#   → Some time periods / conditions get worse predictions
#   → Infrastructure decisions based on bad predictions harm those communities
#   → The council may not even know the model is underperforming for them
#
#   This audit tests model fairness across 4 dimensions:
#   1. Time of day   (rush hour vs overnight)
#   2. Day type      (weekday vs weekend)
#   3. Season        (summer vs winter)
#   4. Weather       (clear vs adverse)

# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocessing import preprocess


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12,
        min_samples_split=10, max_features="sqrt",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# AUDIT FUNCTION — core logic
# =============================================================================

def audit_subgroup(name, mask, y_true, y_pred, results):
    """
    Evaluates model performance on a specific subgroup defined by mask.

    WHY THIS MATTERS:
    A model with overall R²=0.93 might have R²=0.70 for overnight hours
    and R²=0.96 for rush hours. If city council uses this for night-time
    road maintenance scheduling, they are making decisions on bad predictions.
    That is a fairness failure even though the headline number looks great.
    """
    if mask.sum() == 0:
        return

    y_sub   = y_true[mask]
    p_sub   = y_pred[mask]
    mae     = mean_absolute_error(y_sub, p_sub)
    rmse    = np.sqrt(mean_squared_error(y_sub, p_sub))
    r2      = r2_score(y_sub, p_sub)
    count   = mask.sum()

    results.append({
        "Subgroup": name,
        "Count"   : count,
        "MAE"     : round(mae, 1),
        "RMSE"    : round(rmse, 1),
        "R²"      : round(r2, 4)
    })


# =============================================================================
# RUN FULL AUDIT
# =============================================================================

def run_bias_audit(model, X_test, y_test):
    """
    Runs fairness checks across 4 dimensions.
    Each dimension represents a real-world group of road users
    or a scenario that city planners care about.
    """
    predictions = model.predict(X_test)
    results     = []

    # Overall baseline — for comparison
    audit_subgroup("OVERALL", np.ones(len(y_test), dtype=bool),
                   y_test.values, predictions, results)

    # ── DIMENSION 1: Time of Day ──────────────────────────────
    # Rush hour predictions drive the most expensive infrastructure decisions
    # Overnight accuracy matters for maintenance scheduling
    hour = X_test["hour"]
    audit_subgroup("Morning Rush (6–9am)",  ((hour >= 6)  & (hour <= 9)).values,  y_test.values, predictions, results)
    audit_subgroup("Evening Rush (4–7pm)",  ((hour >= 16) & (hour <= 19)).values, y_test.values, predictions, results)
    audit_subgroup("Daytime (10am–3pm)",    ((hour >= 10) & (hour <= 15)).values, y_test.values, predictions, results)
    audit_subgroup("Overnight (11pm–5am)",  ((hour >= 23) | (hour <= 5)).values,  y_test.values, predictions, results)

    # ── DIMENSION 2: Day Type ─────────────────────────────────
    # Weekday vs weekend patterns differ enormously
    # Fairness question: is the model equally reliable for both?
    is_wknd = X_test["is_weekend"]
    audit_subgroup("Weekday",  (is_wknd == 0).values, y_test.values, predictions, results)
    audit_subgroup("Weekend",  (is_wknd == 1).values, y_test.values, predictions, results)

    # ── DIMENSION 3: Season ───────────────────────────────────
    # Minnesota has extreme winters — does the model handle them fairly?
    # Training data: 2012–2018. If climate has shifted, winter predictions may drift.
    month = X_test["month"]
    audit_subgroup("Summer (Jun–Aug)",  ((month >= 6)  & (month <= 8)).values,  y_test.values, predictions, results)
    audit_subgroup("Winter (Dec–Feb)",  ((month == 12) | (month <= 2)).values,  y_test.values, predictions, results)
    audit_subgroup("Spring (Mar–May)",  ((month >= 3)  & (month <= 5)).values,  y_test.values, predictions, results)
    audit_subgroup("Autumn (Sep–Nov)",  ((month >= 9)  & (month <= 11)).values, y_test.values, predictions, results)

    # ── DIMENSION 4: Weather Conditions ───────────────────────
    # Adverse weather = higher stakes for road safety decisions
    # If model is less accurate in bad weather, that is a safety risk
    # We use temperature as a proxy (Kelvin: >285K ≈ warm, <270K ≈ cold/snow)
    temp = X_test["temp"]
    audit_subgroup("Warm Weather (>285K)",  (temp > 0.3).values,   y_test.values, predictions, results)
    audit_subgroup("Cold Weather (<270K)",  (temp < -0.5).values,  y_test.values, predictions, results)

    return pd.DataFrame(results), predictions


# =============================================================================
# FLAG BIAS
# =============================================================================

def flag_bias(df, threshold_mae=150, threshold_r2=0.05):
    """
    Automatically flags subgroups where the model underperforms.

    Flagging criteria:
    1. MAE more than `threshold_mae` above overall MAE
       → model makes significantly larger errors for this group
    2. R² more than `threshold_r2` below overall R²
       → model explains significantly less variance for this group

    These thresholds are conservative — in a real deployment,
    the city council would set these based on acceptable risk levels.
    """
    overall_mae = df[df["Subgroup"] == "OVERALL"]["MAE"].values[0]
    overall_r2  = df[df["Subgroup"] == "OVERALL"]["R²"].values[0]

    flags = []
    for _, row in df.iterrows():
        if row["Subgroup"] == "OVERALL":
            continue
        reasons = []
        if row["MAE"] > overall_mae + threshold_mae:
            reasons.append(f"MAE {row['MAE']:.0f} vs overall {overall_mae:.0f} (+{row['MAE']-overall_mae:.0f})")
        if row["R²"] < overall_r2 - threshold_r2:
            reasons.append(f"R² {row['R²']:.4f} vs overall {overall_r2:.4f} (-{overall_r2-row['R²']:.4f})")
        if reasons:
            flags.append({"Subgroup": row["Subgroup"], "Issues": " | ".join(reasons)})

    return pd.DataFrame(flags) if flags else pd.DataFrame(columns=["Subgroup", "Issues"])


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_audit_results(df, flags):
    print("\n" + "=" * 70)
    print("ETHICAL AI — BIAS & FAIRNESS AUDIT REPORT")

    print("=" * 70)
    print(f"\n{'Subgroup':<30} {'Count':>7} {'MAE':>8} {'RMSE':>8} {'R²':>8}  {'Flag'}")
    print("-" * 70)

    flagged = flags["Subgroup"].tolist() if not flags.empty else []
    overall_r2 = df[df["Subgroup"] == "OVERALL"]["R²"].values[0]

    for _, row in df.iterrows():
        flag_str = "⚠ BIAS DETECTED" if row["Subgroup"] in flagged else "✓"
        marker   = "►" if row["Subgroup"] == "OVERALL" else " "
        print(f"{marker} {row['Subgroup']:<28} {row['Count']:>7} {row['MAE']:>8} "
              f"{row['RMSE']:>8} {row['R²']:>8}  {flag_str}")

    print("\n" + "=" * 70)
    if flags.empty:
        print("✓ No significant bias detected across tested subgroups.")
        print(f"  All subgroups perform within acceptable range of overall R²={overall_r2:.4f}")
    else:
        print(f"⚠ BIAS FLAGS ({len(flags)} subgroup(s) require attention):")
        for _, f in flags.iterrows():
            print(f"  • {f['Subgroup']}: {f['Issues']}")
        print("\n  RECOMMENDATION: Do not use model predictions for these subgroups")
        print("  without additional validation or human oversight.")
    print("=" * 70)


# =============================================================================
# PLOT AUDIT RESULTS
# =============================================================================

def plot_audit(df):
    """
    Visual audit chart for the video presentation and report.
    Red bars = underperforming subgroups (potential bias).
    Green bars = performing at or above overall baseline.
    """
    overall_r2 = df[df["Subgroup"] == "OVERALL"]["R²"].values[0]
    sub_df     = df[df["Subgroup"] != "OVERALL"].copy()

    colors = ["tomato" if r < overall_r2 - 0.05 else "seagreen"
              for r in sub_df["R²"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Ethical AI — Fairness Audit: Random Forest Traffic Predictor\n",
                 fontsize=13, fontweight='bold')

    # Plot 1: R² by subgroup
    bars = axes[0].barh(sub_df["Subgroup"], sub_df["R²"],
                        color=colors, alpha=0.85)
    axes[0].axvline(overall_r2, color='navy', linewidth=2,
                    linestyle='--', label=f'Overall R²={overall_r2:.4f}')
    axes[0].axvline(overall_r2 - 0.05, color='red', linewidth=1,
                    linestyle=':', label='Bias threshold (−0.05)')
    axes[0].set_xlabel("R² Score")
    axes[0].set_title("R² Score by Subgroup\n(red = below bias threshold)")
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0.5, 1.0)

    # Plot 2: MAE by subgroup
    overall_mae = df[df["Subgroup"] == "OVERALL"]["MAE"].values[0]
    mae_colors  = ["tomato" if m > overall_mae + 150 else "seagreen"
                   for m in sub_df["MAE"]]
    axes[1].barh(sub_df["Subgroup"], sub_df["MAE"],
                 color=mae_colors, alpha=0.85)
    axes[1].axvline(overall_mae, color='navy', linewidth=2,
                    linestyle='--', label=f'Overall MAE={overall_mae:.0f}')
    axes[1].axvline(overall_mae + 150, color='red', linewidth=1,
                    linestyle=':', label='Bias threshold (+150)')
    axes[1].set_xlabel("MAE (vehicles/hour)")
    axes[1].set_title("Mean Absolute Error by Subgroup\n(red = above bias threshold)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("bias_audit_results.png", dpi=150)
    plt.show()
    print("\nPlot saved: bias_audit_results.png")
    print("(Include this chart in your Assignment 3 report and video)")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("Loading data and training model...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()
    model = train_model(X_train, y_train)

    print("Running bias audit on test set...")
    audit_df, predictions = run_bias_audit(model, X_test, y_test)

    flags = flag_bias(audit_df)
    print_audit_results(audit_df, flags)
    plot_audit(audit_df)

    print("\n✓ Save bias_audit_results.png for your report and video.")
    print("✓ Next: run transparency.py to explain individual predictions.")
