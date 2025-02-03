import pandas as pd
import numpy as np

pd.read_csv(
    "data/ccpp/ccpp_data.csv",
    header=0,
    usecols=["AT", "AP", "RH", "PE"],
).to_csv("data/ccpp/sample.csv", index=False)
