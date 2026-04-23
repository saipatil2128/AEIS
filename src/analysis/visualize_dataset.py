import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")

df = pd.read_csv("data/processed/final_hybrid_dataset.csv")

print("Dataset shape:", df.shape)


# ---------------------------------------------------
# Sample dataset for visualization (faster + readable)
# ---------------------------------------------------

df_sample = df.sample(100000, random_state=42)


# ---------------------------------------------------
# Balanced dataset for class distribution visualization
# ---------------------------------------------------

normal = df[df["label"] == 0]
attack = df[df["label"] == 1].sample(len(normal), random_state=42)

balanced_vis = pd.concat([normal, attack])


# ---------------------------------------------------
# 1️⃣ Class Distribution (Balanced Visualization)
# ---------------------------------------------------

plt.figure()

balanced_vis["label"].value_counts().plot(kind="bar")

plt.title("Balanced Normal vs Attack Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.xticks([0,1], ["Normal","Attack"])

plt.tight_layout()

plt.savefig("class_distribution.png", dpi=300)

plt.show()


# ---------------------------------------------------
# 2️⃣ Feature Correlation Heatmap
# ---------------------------------------------------

plt.figure()

corr = df_sample.drop(columns=["label"]).corr()

sns.heatmap(corr, annot=True)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig("feature_correlation_heatmap.png", dpi=300)

plt.show()


# ---------------------------------------------------
# 3️⃣ Packets per Minute Distribution
# ---------------------------------------------------

plt.figure()

sns.histplot(df_sample["packets_per_min"], bins=50, log_scale=True)

plt.title("Distribution of Packets per Minute")
plt.xlabel("Packets per Minute (log scale)")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("packets_per_min_distribution.png", dpi=300)

plt.show()


print("Visualization completed and images saved.")