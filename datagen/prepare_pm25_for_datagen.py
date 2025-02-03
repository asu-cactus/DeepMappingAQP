import pandas as pd

df = pd.read_csv(
    "../data/pm25/PRSA_data.csv", header=0, usecols=["pm2.5", "DEWP", "TEMP", "PRES"]
)
df = df.dropna()
df.to_csv("../data/pm25/sample.csv", index=False)
