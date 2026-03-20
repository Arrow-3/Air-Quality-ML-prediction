import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_analysis(data):

    print("=== Exploratory Data Analysis ===")

    # -------------------------------
    # 1. BASIC INFO
    # -------------------------------
    print("\nDataset Info:")
    print(data.info())

    print("\nStatistical Summary:")
    print(data.describe())

    # -------------------------------
    # 2. CORRELATION HEATMAP
    # -------------------------------
    plt.figure(figsize=(12, 8))

    corr = data.corr()

    sns.heatmap(
        corr,
        annot=False,
        cmap="viridis",
        vmin=-1,
        vmax=1
    )

    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()

    # -------------------------------
    # 3. SCATTER PLOTS (FEATURE vs RH)
    # -------------------------------
    features = data.columns.drop(['RH'])

    n_cols = 3
    n_rows = (len(features) // n_cols) + 1

    plt.figure(figsize=(15, 10))

    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)

        plt.scatter(data[feature], data['RH'], s=10)

        plt.xlabel(feature)
        plt.ylabel("RH")
        plt.title(f"{feature} vs RH")

    plt.tight_layout()
    plt.suptitle("Feature vs RH Relationships", y=1.02)
    plt.savefig("scatter_plots.png")
    plt.show()
