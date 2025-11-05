import pandas as pd

pd.read_csv(
    "data/tpc-ds/store_sales.dat",
    header=None,
    sep="|",
    usecols=[11, 12],
    names=["wholesale_cost", "list_price"],
).dropna().to_csv("data/tpc-ds/sample.csv", index=False)

# pd.read_csv(
#     "data/catalog_returns/catalog_returns.dat",
#     header=None,
#     sep="|",
#     usecols=[22, 26],
#     names=["return_ship_cost", "net_loss"],
# ).dropna().to_csv("data/catalog_returns/sample.csv", index=False)

pd.read_csv(
    "data/part/part.tbl",
    sep="|",
    usecols=[0, 7],
    names=["partkey", "retailprice"],
).dropna().to_csv("data/part/sample.csv", index=False)

pd.read_csv(
    "data/lineitem/lineitem.tbl",
    sep="|",
    usecols=[4, 5],
    names=["quantity", "extendedprice"],
).dropna().to_csv("data/lineitem/sample.csv", index=False)
