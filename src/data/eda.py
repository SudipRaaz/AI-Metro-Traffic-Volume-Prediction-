import pandas as pd
import matplotlib.pyplot as plt


def basic_info(df: pd.DataFrame):
    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())


def check_missing_values(df: pd.DataFrame):
    print("\nMissing Values:")
    print(df.isnull().sum())


def plot_histograms(df: pd.DataFrame):
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()


def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap")
    plt.show()


def target_distribution(y: pd.Series):
    plt.figure()
    plt.hist(y)
    plt.title("Target Distribution")
    plt.xlabel("Target")
    plt.ylabel("Frequency")
    plt.show()
    
def eda_insights(df):
    print("\n=== KEY INSIGHTS ===")

    print("\nCorrelation with target:")
    print(df.corr(numeric_only=True)["traffic_volume"].sort_values(ascending=False))

    print("\nTop 5 busiest hours:")
    print(df.groupby("hour")["traffic_volume"].mean().sort_values(ascending=False).head())

    print("\nWeekend vs Weekday:")
    print(df.groupby("is_weekend")["traffic_volume"].mean())


def run_eda(X: pd.DataFrame, y: pd.Series):
    df = pd.concat([X, y], axis=1)

    basic_info(df)
    check_missing_values(df)

    plot_histograms(df)
    correlation_heatmap(df)
    target_distribution(y)