import pandas as pd

def merge_datasets(local_dataset, nbaiot_dataset):

    combined = pd.concat([local_dataset, nbaiot_dataset], ignore_index=True)

    print("Merged dataset shape:", combined.shape)

    return combined

def save_dataset(df, path):

    df.to_csv(path, index=False)

    print("Dataset saved:", path)