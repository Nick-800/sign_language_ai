import pandas as pd
import glob
import os

all_data = []
for file in glob.glob("dataset/*.csv"):
    df = pd.read_csv(file, header=None)
    df["label"] = os.path.basename(file).replace(".csv", "")
    all_data.append(df)

df = pd.concat(all_data)
df.to_csv("gesture_dataset.csv", index=False)
print("✅ Dataset saved as gesture_dataset.csv")
