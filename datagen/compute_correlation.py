import pandas as pd

dep_dict = {
    "tpc-ds": ["list_price", "wholesale_cost"],
    "flights": ["DISTANCE", "TAXI_OUT"],
    "pm25": ["PRES", "pm25"],
    "ccpp": ["RH", "PE"],
}

for data_name, usecols in dep_dict.items():

    df1 = pd.read_csv(f"data/{data_name}/sample.csv", usecols=usecols)
    correlation = df1.corr()
    print(f"Original data correlation:\n{correlation}")

    df2 = pd.read_csv(f"data/{data_name}/dataset_sum.csv", usecols=usecols)
    correlation = df2.corr()
    print(f"Generated data correlation:\n{correlation}")
