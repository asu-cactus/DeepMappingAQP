import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/pm25/PRSA_data.csv",
    header=0,
    usecols=["pm2.5", "DEWP", "TEMP", "PRES"],
)
df = df.rename(columns={"pm2.5": "pm25"})
df = df.dropna()
df.astype(np.int32).to_csv("data/pm25/sample.csv", index=False)
