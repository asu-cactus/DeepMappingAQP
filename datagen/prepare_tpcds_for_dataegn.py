import pandas as pd
import numpy as np

pd.read_csv(
    "data/tpc-ds/store_sales.dat",
    header=None,
    sep="|",
    usecols=[11, 12],
    names=["wholesale_cost", "list_price"],
).to_csv("data/tpc-ds/sample.csv", index=False)
