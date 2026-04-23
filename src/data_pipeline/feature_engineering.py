import pandas as pd

input_file = "data/processed/local_cleaned.csv"
output_file = "data/processed/local_features.csv"

print("Loading cleaned dataset ...")

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

#Comvert timestamps
df["frame.time"] = pd.to_datetime(df["frame.time"])

#Rename columns for better readability
df = df.rename(columns={
    "ip.src": "device_ip",
    "ip.dst": "dest_ip",
    "frame.len": "packet_size",
    "_ws.col.protocol": "protocol"
})

#Remove rows with unknown device
df = df[df["device_ip"] != "unknown"]

#Create time window (1 minute)
df["time_window"] = df["frame.time"].dt.floor("min")

#Extract hour feature
df["hour"] = df["frame.time"].dt.hour

#Behavioral aggregations
features = df.groupby(["device_ip", "time_window"]).agg(
    packets_per_min = ("packet_size", "count"),
    avg_packet_size = ("packet_size", "mean"),
    unique_destinations = ("dest_ip", "nunique"),
).reset_index()

#Add hour feature
features["hour"] = features["time_window"].dt.hour

print("Feature dataset shape:", features.shape)

features.to_csv(output_file, index=False)

print("Saved feature dataset:", output_file)