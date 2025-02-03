import pandas as pd
import numpy as np
import math
import pdb

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def read_data(data_name):
    if data_name == "store_sales":

        df = pd.read_csv(
            "data/tpc-ds/store_sales.dat",
            sep="|",
            usecols=[10, 13, 22],
            names=["quantity", "sales_price", "net_profit"],
        )
    elif data_name == "flights":
        df = pd.read_csv(
            "data/flights/sample.csv",
            header=0,
            usecols=["UNIQUE_CARRIER", "DEST_STATE_ABR", "TAXI_OUT", "DISTANCE"],
        )
    elif data_name == "pm25":
        df = pd.read_csv(
            "data/pm25/PRSA_data.csv",
            header=0,
            usecols=["pm2.5", "DEWP", "TEMP", "PRES"],
        )
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    pdb.set_trace()
    return df


def bin_and_cumsum(df, indep, dep, resolution):
    df = df[[indep, dep]]
    df = df.dropna()

    indep_min = df[indep].min()
    indep_max = df[indep].max()

    num_max = math.ceil(indep_max / resolution)
    num_min = math.floor(indep_min / resolution)
    print(
        f"""
    min indep: {indep_min}, max indep: {indep_max}
    range min: {num_min * resolution}, range max: {num_max * resolution}
    """
    )
    bin_edges = [i * resolution for i in range(num_min, num_max + 1)]

    # Sum df["dep"] for each bin
    bin_sum = df.groupby(pd.cut(df[indep], bin_edges))[dep].sum()
    cum_sum = bin_sum.cumsum()

    return bin_edges, cum_sum


def prepare_training_data(
    df: pd.DataFrame,
    indep: str,
    dep: str,
    resolution: float,
    groupby: str = None,
    selection_key: list[str] = None,
    selections: list[str] = None,
):
    print(f"Resolution: {resolution}")
    if selections is not None and selection_key is not None:
        for selection, key in zip(selections, selection_key):
            df = df[df[key] == selection]
    if groupby is not None:
        Xs = []
        ys = []
        unique_keys = df[groupby].unique()
        for key in unique_keys:
            print(f"Group: {key}")
            df_group = df[df[groupby] == key]

            bin_edges, cum_sum = bin_and_cumsum(df_group, indep, dep, resolution)
            Xs.append(np.array(bin_edges[1:]))
            ys.append(np.array(cum_sum))
        pdb.set_trace()
        return Xs, ys

    bin_edges, cum_sum = bin_and_cumsum(df, indep, dep, resolution)

    pdb.set_trace()
    X = np.array(bin_edges[1:])
    y = np.array(cum_sum)
    assert len(X) == len(y)
    return X, y


if __name__ == "__main__":
    df = read_data("pm25")
    # df = read_data("flights")
    # prepare_training_data(
    #     df,
    #     "DISTANCE",
    #     "TAXI_OUT",
    #     100,
    #     groupby="DEST_STATE_ABR",
    #     selection_key=["UNIQUE_CARRIER"],
    #     selections=["UA"],
    # )
