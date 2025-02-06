import pandas as pd
import numpy as np
import torch
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pdb

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def read_data(data_name: str) -> pd.DataFrame:
    if data_name == "store_sales":

        df = pd.read_csv(
            "data/tpc-ds/store_sales.dat",
            sep="|",
            usecols=[10, 11, 12, 13, 22],
            names=[
                "quantity",
                "wholesale_cost",
                "list_price",
                "sales_price",
                "net_profit",
            ],
        )
    elif data_name == "flights":
        df = pd.read_csv(
            "data/flights/dataset.csv",
            header=0,
            usecols=["TAXI_OUT", "DISTANCE"],
        )
    elif data_name == "pm25":
        df = pd.read_csv(
            "data/pm25/dataset.csv",
            header=0,
            usecols=["pm25", "DEWP", "TEMP", "PRES"],
        )
    elif data_name == "ccpp":
        df = pd.read_csv(
            "data/ccpp/dataset.csv",
            header=0,
            usecols=["AT", "AP", "RH", "PE"],
        )
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

    return df


def bin_and_cumsum(df, indep, dep, resolution):
    df = df[[indep, dep]]
    df = df.dropna()

    indep_min = df[indep].min()
    indep_max = df[indep].max()

    num_max = math.ceil(indep_max / resolution)
    if indep_min % resolution == 0:
        num_min = math.floor(indep_min / resolution) - 1
    else:
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


def run_groupby(df, groupby, indep, dep, resolution):
    Xs = []
    ys = []
    unique_keys = df[groupby].unique()
    for key in unique_keys:
        print(f"Group: {key}")
        df_group = df[df[groupby] == key]

        bin_edges, cum_sum = bin_and_cumsum(df_group, indep, dep, resolution)
        Xs.append(np.array(bin_edges[1:]))
        ys.append(np.array(cum_sum))
    return Xs, ys


def prepare_training_data(
    df: pd.DataFrame,
    indep: str,
    dep: str,
    resolution: float,
    output_scale: float,
    # groupby: str = None,
    # selection: tuple[list[str], list[str]] = None,
    do_standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, MinMaxScaler]:
    print(f"Resolution: {resolution}")
    # if selection is not None:
    #     selection_key, selections = selection
    #     for selection, key in zip(selections, selection_key):
    #         df = df[df[key] == selection]

    bin_edges, cum_sum = bin_and_cumsum(df, indep, dep, resolution)

    X = np.array(bin_edges[1:]).reshape(-1, 1)
    y = np.array(cum_sum).reshape(-1, 1)
    assert len(X) == len(y)
    if do_standardize:
        return standardize_data(X, y, output_scale)
    else:
        return X, y


def standardize_data(
    X: np.array, y: np.array, output_scale: float
) -> tuple[np.array, np.array, StandardScaler, MinMaxScaler]:
    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler(feature_range=(-output_scale, output_scale))
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)
    return X, y, X_scaler, y_scaler


def get_dataloader(X, y, batch_size):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True
    )
    return dataloader


if __name__ == "__main__":
    df = read_data("store_sales")

    # df = read_data("ccpp")

    # df = read_data("pm25")
    # prepare_training_data(df, "TEMP", "pm2.5", 1)

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
