import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Random shuffle rows")
parser.add_argument("--data_name", type=str, required=True, help="Data name")
args = parser.parse_args()
data_name = args.data_name

df = pd.read_csv(f"data/{data_name}/dataset_sum.csv")
# Random shuffle rows and save to a new file
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv(f"data/{data_name}/dataset_sum.csv", index=False)
