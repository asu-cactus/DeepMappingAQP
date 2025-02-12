import pandas as pd


def convert_flights():
    df = pd.read_csv("data/flights/dataset_sum.csv", usecols=["DISTANCE", "TAXI_OUT"])
    # Set all TAXI_OUT values to 1.0
    df["TAXI_OUT"] = 1.0

    df.to_csv("data/flights/dataset_count.csv", index=False)


def convert_ccpp():
    df = pd.read_csv("data/ccpp/dataset_sum.csv")
    # Set all PE values to 1.0
    df["PE"] = 1.0
    df.to_csv("data/ccpp/dataset_count.csv", index=False)


def convert_pm25():
    df = pd.read_csv("data/pm25/dataset_sum.csv")
    # Set all pm2.5 values to 1.0
    df["pm25"] = 1.0
    df.to_csv("data/pm25/dataset_count.csv", index=False)


def convert_tpcds():
    df = pd.read_csv("data/tpc-ds/dataset_sum.csv")
    # Set all pm2.5 values to 1.0
    df["wholesale_cost"] = 1.0
    df.to_csv("data/tpc-ds/dataset_count.csv", index=False)


convert_tpcds()
