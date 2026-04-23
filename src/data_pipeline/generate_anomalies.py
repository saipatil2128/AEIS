import pandas as pd
import numpy as np

input_file = "data/processed/local_features.csv"
output_file = "data/processed/hybrid_dataset.csv"

print("Loading feature dataset...")

df = pd.read_csv(input_file)

print("Normal samples:", df.shape)

# Label normal traffic
df["label"] = 0

# Create anomaly copy
anomalies = df.copy()

# Inject abnormal behavior
anomalies["packets_per_min"] = anomalies["packets_per_min"] * np.random.randint(5,15)
anomalies["unique_destinations"] = anomalies["unique_destinations"] * np.random.randint(5,10)
anomalies["avg_packet_size"] = anomalies["avg_packet_size"] * np.random.uniform(0.5,2.0)

# Label anomalies
anomalies["label"] = 1

print("Anomaly samples:", anomalies.shape)

# Combine datasets
hybrid = pd.concat([df, anomalies])

print("Final dataset:", hybrid.shape)

hybrid.to_csv(output_file, index=False)

print("Hybrid dataset saved:", output_file)