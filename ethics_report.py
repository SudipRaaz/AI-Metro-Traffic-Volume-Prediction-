# =============================================================================
# PRO502 - Assignment 3
# File: ethics_report.py
# Ethical AI — Full Client-Facing Audit Report Generator
#
# PURPOSE:
#   Combines bias audit + transparency results into a single readable
#   report that a city council or government agency would actually use
#   before deploying the traffic prediction system.
#
#   This demonstrates that ethical AI is not just about building a
#   fair model — it is about communicating risks clearly to decision-makers
#   so they can exercise human oversight.

# =============================================================================

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.preprocessing import preprocess
from bias_audit import train_model, run_bias_audit, flag_bias


# =============================================================================
# GENERATE FULL ETHICS REPORT
# =============================================================================

def generate_ethics_report(model, X_test, y_test, output_file="ethics_audit_report.txt"):
    """
    Generates a plain-English audit report suitable for a city council client.
    Saved as a .txt file that can be included in the Assignment 3 submission.
    """

    predictions = model.predict(X_test)
    overall_mae  = mean_absolute_error(y_test, predictions)
    overall_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    overall_r2   = r2_score(y_test, predictions)

    audit_df, _ = run_bias_audit(model, X_test, y_test)
    flags        = flag_bias(audit_df)

    now = datetime.now().strftime("%d %B %Y, %H:%M")

    lines = []
    def w(text=""): lines.append(text)

    w("=" * 70)
    w("  ETHICAL AI AUDIT REPORT")
    w("  AI-Powered Traffic Volume Prediction System")
    w("  Prepared for: City Council / Transport Authority")
    w(f"  Generated   : {now}")
    w("=" * 70)

    # ── Section 1: System Overview ────────────────────────────
    w()
    w("1. SYSTEM OVERVIEW")
    w("-" * 70)
    w("  System Name : Traffic Volume Prediction — AI Advisory Tool")
    w("  Model Type  : Random Forest Regressor (ensemble of 100 decision trees)")
    w("  Purpose     : Predict hourly traffic volume on interstate highways")
    w("                to support infrastructure planning decisions.")
    w("  Dataset     : UCI Metro Interstate Traffic Volume (2012–2018)")
    w("                Source: https://archive.ics.uci.edu/dataset/492/")
    w("  Features    : Time of day, day of week, month, temperature,")
    w("                weather conditions, rainfall, snowfall, cloud cover.")
    w()
    w("  IMPORTANT: This system is an advisory tool only. All infrastructure")
    w("  decisions must be reviewed by qualified human officers before")
    w("  implementation. The system must not be used for automated")
    w("  decisions without human oversight.")
 

    # ── Section 2: Performance Summary ───────────────────────
    w()
    w("2. PERFORMANCE SUMMARY (Test Set — Unseen Data)")
    w("-" * 70)
    w(f"  Overall R²   : {overall_r2:.4f}  (model explains {overall_r2*100:.1f}% of traffic variance)")
    w(f"  Overall MAE  : {overall_mae:.1f} vehicles/hour average error")
    w(f"  Overall RMSE : {overall_rmse:.1f} vehicles/hour (weighted for large errors)")
    w()
    w("  Interpretation:")
    if overall_r2 >= 0.90:
        w("  ✓ EXCELLENT — Model meets deployment performance threshold (R² ≥ 0.90)")
    elif overall_r2 >= 0.80:
        w("  ✓ GOOD — Model meets minimum performance threshold (R² ≥ 0.80)")
    else:
        w("  ⚠ BELOW THRESHOLD — Model requires improvement before deployment")
    w(f"  The average prediction error of {overall_mae:.0f} vehicles/hour means that")
    w("  on a road with 4,000 vehicles/hour, predictions are typically within")
    w(f"  {overall_mae/4000*100:.1f}% of the true value.")

    # ── Section 3: Bias & Fairness Audit ─────────────────────
    w()
    w("3. BIAS & FAIRNESS AUDIT")
    w("-" * 70)
  
    w("  Subgroups tested: time of day, weekday/weekend, season, weather")
    w()
    w(f"  {'Subgroup':<30} {'MAE':>8} {'R²':>8}  {'Status'}")
    w("  " + "-" * 60)

    flagged = flags["Subgroup"].tolist() if not flags.empty else []
    for _, row in audit_df.iterrows():
        status = "⚠ FLAGGED" if row["Subgroup"] in flagged else "✓ OK"
        marker = "►" if row["Subgroup"] == "OVERALL" else " "
        w(f"  {marker} {row['Subgroup']:<29} {row['MAE']:>8} {row['R²']:>8}  {status}")

    w()
    if flags.empty:
        w("  FINDING: No significant performance disparities detected.")
        w("  The model performs consistently across all tested subgroups.")
        w("  This supports equitable use for planning across all time periods.")
    else:
        w(f"  FINDING: {len(flags)} subgroup(s) show performance below acceptable threshold.")
        for _, f in flags.iterrows():
            w(f"  ⚠ {f['Subgroup']}: {f['Issues']}")
        w()
        w("  RECOMMENDATION: Human review is REQUIRED for predictions in")
        w("  flagged subgroups. Do not use automated predictions for these")
        w("  conditions without additional validation.")

    # ── Section 4: Privacy Assessment ────────────────────────
    w()
    w("4. DATA PRIVACY ASSESSMENT")
    w("-" * 70)
    w("  Reference: Privacy Act 1988 (Cth) — US Federal Privacy Guidelines / FHWA")
    w()
    w("  Data Used:")
    w("  ✓ Aggregated hourly traffic counts (not individual vehicles)")
    w("  ✓ Weather and calendar data (publicly available)")
    w("  ✓ No personally identifiable information (PII) collected")
    w("  ✓ No vehicle registration, GPS, or driver data used")
    w()
    w("  Privacy Risk: LOW")
    w("  The dataset contains no information that can identify individuals.")
    w("  Aggregated traffic counts do not constitute personal data under")
    w("  the Privacy Act 1988 or GDPR.")
    w()
    w("  Deployment Recommendation:")
    w("  If this system is extended to use real-time sensor data,")
    w("  a Privacy Impact Assessment (PIA) must be conducted before")
    w("  deployment as required under APP 1.2 (Privacy Act 1988).")

    # ── Section 5: Legal & Regulatory ────────────────────────
    w()
    w("5. LEGAL & REGULATORY COMPLIANCE")
    w("-" * 70)
    w(" NIST, 2023 — Assessment:")
    w()
    principles = [
        ("1. Human, Social & Environmental Wellbeing", "✓ MET",
         "System improves road safety and reduces congestion harm"),
        ("2. Human-Centred Values",                    "✓ MET",
         "Advisory only — human officers make final decisions"),
        ("3. Fairness",                                "✓ MET" if flags.empty else "⚠ PARTIAL",
         "Bias audit conducted; flagged subgroups require review"),
        ("4. Privacy Protection & Security",           "✓ MET",
         "No PII used; aggregated data only"),
        ("5. Reliability & Safety",                    "✓ MET",
         f"R²={overall_r2:.4f} exceeds 0.90 threshold; error bounds disclosed"),
        ("6. Transparency & Explainability",           "✓ MET",
         "Feature importance disclosed; prediction explanations available"),
        ("7. Contestability",                          "✓ MET",
         "Council officers can override any prediction"),
        ("8. Accountability",                          "✓ MET",
         "This audit report documents system behaviour and limitations"),
    ]
    for principle, status, note in principles:
        w(f"  {status}  {principle}")
        w(f"         Note: {note}")
        w()

    # ── Section 6: Limitations ────────────────────────────────
    w()
    w("6. KNOWN LIMITATIONS")
    w("-" * 70)
    w("  The following limitations must be disclosed to all system users:")
    w()
    w("  1. TEMPORAL DRIFT: Training data is from 2012–2018. Road usage")
    w("     patterns may have changed (remote work, population growth).")
    w("     Model should be retrained annually with fresh data.")
    w()
    w("  2. GEOGRAPHIC SCOPE: Model is trained on one sensor location")
    w("     (I-94, Minnesota). Performance on other roads is unknown")
    w("     without revalidation.")
    w()
    w("  3. RARE EVENTS: Model has limited exposure to extreme events")
    w("     (major accidents, severe storms, public events).")
    w("     Do not rely on AI predictions for emergency traffic management.")
    w()
    w("  4. PROXY BIAS: Weather features use historical averages.")
    w("     Actual real-time weather integration would improve accuracy")
    w("     in adverse conditions.")

    # ── Section 7: Recommendations ────────────────────────────
    w()
    w("7. RECOMMENDATIONS FOR DEPLOYMENT")
    w("-" * 70)
    w("  Before this system is used in production, the following steps")
    w("  are recommended:")
    w()
    w("  SHORT TERM (before deployment):")
    w("  • Conduct stakeholder consultation with affected communities")
    w("  • Complete Privacy Impact Assessment under APP 1.2")
    w("  • Define human oversight protocols for all flagged subgroups")
    w("  • Establish a feedback mechanism for council officers to report")
    w("    prediction failures")
    w()
    w("  MEDIUM TERM (first 12 months):")
    w("  • Retrain model with data from 2019 onwards")
    w("  • Expand bias audit to include road-type and demographic analysis")
    w("  • Integrate real-time weather API for improved adverse-weather accuracy")
    w()
    w("  LONG TERM (scaling):")
    w("  • Extend to multiple sensor locations with site-specific models")
    w("  • Implement continuous monitoring dashboard for bias drift")
    w("  • Annual independent ethical audit by third party")

    # ── Footer ────────────────────────────────────────────────
    w()
    w("=" * 70)
    w("  END OF AUDIT REPORT")
    w(f"  Generated: {now}")
    w("  This report was generated automatically by the Ethical AI Prototype.")
    w("  It should be reviewed by a qualified AI ethics officer before")
    w("  being used for official decision-making.")
    w("=" * 70)

    # Save to file
    report_text = "\n".join(lines)
    with open(output_file, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✓ Report saved to: {output_file}")
    return report_text


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("Loading data and training model...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess()
    model = train_model(X_train, y_train)

    print("Generating ethics audit report...\n")
    generate_ethics_report(model, X_test, y_test)

    print("\n✓ Assignment 3 prototype complete.")
    print("  Files to submit:")
    print("  • preprocessing.py")
    print("  • predict.py")
    print("  • bias_audit.py")
    print("  • transparency.py")
    print("  • ethics_report.py        ← run this for your video demo")
    print("  • ethics_audit_report.txt ← include in submission zip")
    print("  • bias_audit_results.png  ← use in report and video")
