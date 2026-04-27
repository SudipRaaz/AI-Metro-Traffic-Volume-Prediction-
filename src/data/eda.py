import pandas as pd
import matplotlib.pyplot as plt


def run_eda(X: pd.DataFrame, y: pd.Series):
    df = pd.concat([X, y], axis=1)

    print("\n=== DATA INFO ===")
    print(df.info())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== SUMMARY ===")
    print(df.describe())

    # =========================
    # CORRELATION (TARGET-FOCUSED)
    # =========================
    corr = df.corr(numeric_only=True)

    target_corr = corr["traffic_volume"].sort_values(ascending=False)

    print("\n=== CORRELATION WITH TARGET ===")
    print(target_corr)

    plt.figure()
    target_corr.drop("traffic_volume").plot(kind="bar")
    plt.title("Feature Correlation with Traffic Volume")
    plt.ylabel("Correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"correlation_bar.png", dpi=300, bbox_inches="tight")
    plt.show()

    #Correlation heatmap
    corr = df.corr(numeric_only=True) 
    plt.figure(figsize=(8, 6)) 
    plt.imshow(corr)
    plt.colorbar() 
    plt.title("Correlation Heatmap") 
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig(f"correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    # =========================
    # TIME ANALYSIS (IMPORTANT)
    # =========================
    if "hour" in df.columns:
        plt.figure()
        df.groupby("hour")["traffic_volume"].mean().plot()
        plt.title("Average Traffic Volume by Hour")
        plt.ylabel("Traffic Volume")
        plt.savefig(f"traffic_by_hour.png", dpi=300, bbox_inches="tight")
        plt.show()

    if "day_of_week" in df.columns:
        plt.figure()
        df.groupby("day_of_week")["traffic_volume"].mean().plot()
        plt.title("Traffic Volume by Day of Week")
        plt.savefig(f"traffic_by_week.png", dpi=300, bbox_inches="tight")

        plt.show()

    if "month" in df.columns:
        plt.figure()
        df.groupby("month")["traffic_volume"].mean().plot()
        plt.title("Traffic Volume by Month")
        plt.savefig(f"traffic_by_month.png", dpi=300, bbox_inches="tight")
        plt.show()

def plot_actual_vs_predicted(y_test, y_pred, model_name="Model"):
    plt.figure()

    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.savefig(f"{model_name}_actual_vs_predicted.png", dpi=300, bbox_inches="tight")

    plt.show()