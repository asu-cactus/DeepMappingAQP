import pandas as pd
import numpy as np
import torch
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pdb

import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def read_data(data_name: str, task_type: str) -> pd.DataFrame:
    if data_name == "store_sales":
        df = pd.read_csv(
            f"data/tpc-ds/dataset_{task_type}.csv",
            header=0,
        )

    elif data_name == "flights":
        df = pd.read_csv(
            f"data/flights/dataset_{task_type}.csv",
            header=0,
            usecols=["TAXI_OUT", "DISTANCE", "ARR_DELAY"],
        )
    elif data_name == "pm25":
        df = pd.read_csv(
            f"data/pm25/dataset_{task_type}.csv",
            header=0,
            usecols=["pm25", "DEWP", "TEMP", "PRES"],
        )
    elif data_name == "ccpp":
        df = pd.read_csv(
            f"data/ccpp/dataset_{task_type}.csv",
            header=0,
            usecols=["AT", "AP", "RH", "PE"],
        )
    else:
        raise ValueError(f"Unknown data_name: {data_name}")

    return df


def make_histogram1d(df, indep, dep, resolution, bin_edges=None):
    df = df[[indep, dep]]
    df = df.dropna()

    if bin_edges is None:
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
    # bin_intervals = pd.cut(df[indep], bin_edges, include_lowest=True)
    # bin_sum = df.groupby(bin_intervals, observed=False)[dep].sum()
    # cum_sum = bin_sum.cumsum()
    histogram, _ = np.histogram(df[indep], bins=bin_edges, weights=df[dep])
    return np.asarray(bin_edges), np.asarray(histogram)


def make_histogram2d(
    df, indeps, dep, resolutions, bin_edges=None
) -> tuple[np.ndarray, np.ndarray]:
    if bin_edges is None:

        indep_mins = (df[indeps[0]].min(), df[indeps[1]].min())
        indep_maxs = (df[indeps[0]].max(), df[indeps[1]].max())
        num_maxs = (
            math.ceil(indep_maxs[0] / resolutions[0]),
            math.ceil(indep_maxs[1] / resolutions[1]),
        )
        num_mins = (
            math.floor(indep_mins[0] / resolutions[0]),
            math.floor(indep_mins[1] / resolutions[1]),
        )
        range_mins = (num_mins[0] * resolutions[0], num_mins[1] * resolutions[1])
        range_maxs = (num_maxs[0] * resolutions[0], num_maxs[1] * resolutions[1])
        print(
            f"""
        min indeps: {indep_mins}, max indeps: {indep_maxs}
        range min: {range_mins}, range max: {range_maxs}
        """
        )

        bin_edges_dim1 = [
            i * resolutions[0] for i in range(num_mins[0], num_maxs[0] + 1)
        ]
        bin_edges_dim2 = [
            i * resolutions[1] for i in range(num_mins[1], num_maxs[1] + 1)
        ]

        # Combine bin_edges_dim1 and bin_edges_dim2 as two columns and convert to numpy array
        bin_edges = np.asarray(
            [
                (bin_edge_1, bin_edge_2)
                for bin_edge_1 in bin_edges_dim1
                for bin_edge_2 in bin_edges_dim2
            ]
        )

    # create 2d histogram
    histogram, _, _ = np.histogram2d(
        df[indeps[0]],
        df[indeps[1]],
        bins=[bin_edges[:, 0].flatten(), bin_edges[:, 1].flatten()],
        weights=df[dep],
    )

    # bin_edges = [(bin_edges_dim1[0], bin_edges_dim2[0])] + [
    #     (bin_edge_1, bin_edge_2)
    #     for bin_edge_1 in bin_edges_dim1[1:]
    #     for bin_edge_2 in bin_edges_dim2[1:]
    # ]

    return np.asarray(bin_edges), np.asarray(histogram)


def get_X_and_y(df, indeps, dep, ndim_input, resolutions, bin_edges=None):
    if ndim_input == 1:
        bin_edges, histogram = make_histogram1d(
            df, indeps[0], dep, resolutions[0], bin_edges
        )
        cum_sum = np.cumsum(histogram)
    else:
        bin_edges, histogram = make_histogram2d(df, indeps, dep, resolutions, bin_edges)
        cum_sum = np.cumsum(np.cumsum(histogram, axis=0), axis=1)

    X = bin_edges[1:].reshape(-1, ndim_input)
    y = cum_sum.reshape(-1, 1)

    assert len(X) == len(y)
    return X, y


def prepare_full_data(args) -> tuple[np.ndarray, np.ndarray]:
    df = read_data(args.data_name, args.task_type)
    X_all, y_all = get_X_and_y(
        df, args.indeps, args.dep, args.ndim_input, args.resolutions
    )
    return X_all, y_all


def prepare_training_data(args) -> tuple[np.ndarray, np.ndarray]:
    # Load data if exists:
    data_name = args.data_name
    folder_name = data_name if data_name != "store_sales" else "tpc-ds"
    full_path = (
        f"data/{folder_name}/traindata_{args.ndim_input}D_sr{args.sample_ratio}.npz"
    )
    if os.path.exists(full_path):
        npzfile = np.load(full_path)
        X, y = npzfile["X"], npzfile["y"]
        return X, y

    df = read_data(data_name, args.task_type)

    if args.align:
        if args.ndim_input == 1:
            bin_edges, _ = make_histogram1d(
                df, args.indeps[0], args.dep, args.resolutions[0]
            )
        else:
            bin_edges, _ = make_histogram2d(df, args.indeps, args.dep, args.resolutions)
    else:  # use the bin_edges from the full data
        bin_edges = None

    if args.sample_ratio < 1.0:
        df = df.sample(frac=args.sample_ratio, random_state=42)
    X_train, y_train = get_X_and_y(
        df, args.indeps, args.dep, args.ndim_input, args.resolutions, bin_edges
    )

    # Save training data
    np.savez(full_path, X=X_train, y=y_train)
    return X_train, y_train


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
