import os
import pandas as pd

def load_nbaiot_dataset(dataset_path):

    dataframes = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:

            if file.endswith(".csv"):

                file_path = os.path.join(root, file)
                print("Loading:", file_path)

                df = pd.read_csv(file_path)

                # Assign label from filename
                if "benign" in file.lower():
                    df["label"] = 0
                else:
                    df["label"] = 1

                dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    print("Combined dataset shape:", combined.shape)

    return combined