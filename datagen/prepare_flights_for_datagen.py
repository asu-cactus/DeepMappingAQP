import pandas as pd


df = pd.read_csv(
    "data/flights/sample_original.csv",
    header=0,
    usecols=["ARR_DELAY", "TAXI_OUT", "DISTANCE"],
).to_csv("data/flights/sample.csv", index=False)
