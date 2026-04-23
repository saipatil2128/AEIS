import pandas as pd

from data_loader import load_nbaiot_dataset
from preprocessing import process_nbaiot_data
from utils import merge_datasets, save_dataset

nbaiot_path = "data/external/public_dataset"

print("Loading N-BaIoT dataset...")

nbaiot_df = load_nbaiot_dataset(nbaiot_path)

print("Preprocessing...")

nbaiot_features = process_nbaiot_data(nbaiot_df)

nbaiot_features = nbaiot_features.fillna(nbaiot_features.mean(numeric_only=True))

local_dataset = pd.read_csv("data/processed/hybrid_dataset.csv")

final_dataset = merge_datasets(local_dataset, nbaiot_features)

final_dataset = final_dataset.drop(columns=["device_ip", "time_window"], errors="ignore")

final_dataset = final_dataset.fillna(final_dataset.mean(numeric_only=True))

save_dataset(final_dataset, "data/processed/final_hybrid_dataset.csv")