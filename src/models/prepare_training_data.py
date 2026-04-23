import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading dataset...")

df = pd.read_csv("data/processed/final_hybrid_dataset.csv")
# Separate classes
normal = df[df["label"] == 0]
attack = df[df["label"] == 1]

print("Normal samples:", len(normal))
print("Attack samples:", len(attack))

# Downsample attacks to match normal count
attack_sample = attack.sample(n=len(normal)*5, random_state=42)

df_balanced = pd.concat([normal, attack_sample])

print("Balanced dataset shape:", df_balanced.shape)


print("Dataset shape:", df.shape)

#Features and labels
X = df_balanced.drop("label", axis=1)
y = df_balanced["label"]

print("Features shape:", X.shape)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42,
    stratify=y   
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

#Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Save processed data
pd.DataFrame(X_train_scaled).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

#Save scaler for inference
joblib.dump(scaler, "models/scaler.pkl")

print("Training data prepared and saved.")