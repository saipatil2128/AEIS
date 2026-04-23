import pandas as pd

input_file = "data/raw/local/normal_capture_trimmed.csv"
output_file = "data/processed/local_cleaned.csv"

print("Loading dataset...")

df = pd.read_csv(input_file)

print("Original shape:", df.shape)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Convert packet size
df["frame.len"] = pd.to_numeric(df["frame.len"], errors="coerce")

# Convert timestamp
df["frame.time"] = pd.to_datetime(df["frame.time"], errors="coerce")

# Remove bad rows
df = df.dropna(subset=["frame.time", "frame.len"])

# Fill missing IP values
df["ip.src"] = df["ip.src"].fillna("unknown")
df["ip.dst"] = df["ip.dst"].fillna("unknown")

# Fill missing protocol
df["_ws.col.protocol"] = df["_ws.col.protocol"].fillna("unknown")

print("After cleaning:", df.shape)

df.to_csv(output_file, index=False)

print("Saved cleaned dataset:", output_file)