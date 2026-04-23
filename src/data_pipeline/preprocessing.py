import pandas as pd

def process_nbaiot_data(df):

    processed = pd.DataFrame()

    #Approximate mapping
    processed["packets_per_min"] = df["MI_dir_L5_weight"]
    processed["avg_packet_size"] = df["MI_dir_L5_mean"]
    processed["unique_destinations"] = df["MI_dir_L5_variance"]

    #Random hour since dataset has no timestamp
    processed["hour"] = 12

    processed["label"] = 1  #N-BaIoT samples are attacks

    return processed
