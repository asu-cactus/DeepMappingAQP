import math
import pdb
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def read_data(args) -> pd.DataFrame:
    data_name = args.data_name
    task_type = args.task_type
    usecols = [args.dep] + args.indeps
    folder_name = data_name if data_name != "store_sales" else "tpc-ds"
    df = pd.read_csv(
        f"data/{folder_name}/dataset_{task_type}.csv",
        header=0,
        usecols=usecols,
    )
    df = df.dropna()
    return df


def read_insertion_data(args, filename) -> pd.DataFrame:
    data_name = args.data_name
    usecols = [args.dep] + args.indeps
    folder_name = data_name if data_name != "store_sales" else "tpc-ds"
    df = pd.read_csv(
        f"data/update_data/{folder_name}/{filename}",
        header=0,
        usecols=usecols,
    )
    df = df.dropna()
    return df


def make_histogram1d(df, indep, dep, resolution, bin_edges=None):
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
    df = read_data(args)
    X_all, y_all = get_X_and_y(
        df, args.indeps, args.dep, args.ndim_input, args.resolutions
    )
    return X_all, y_all


def prepare_full_data_with_insertion(
    args, do_sample: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    # Only implement 1D input for now
    indep, dep = args.indeps[0], args.dep
    resolution = args.resolutions[0]

    df = read_data(args)
    if do_sample:
        df = df.sample(frac=args.sample_ratio, random_state=42)
    bin_edges, histogram = make_histogram1d(df, indep, dep, resolution)
    cum_sum = np.cumsum(histogram)

    filename = (
        "insert_filtered_by_size.csv"
        if args.uniform_update
        else "insert_filtered_by_range.csv"
    )
    df_insert = read_insertion_data(args, filename)
    ys = [cum_sum.copy()]
    batch_size = int(len(df_insert) / args.n_insert_batch)
    for i in range(args.n_insert_batch):
        insert_batch = df_insert[batch_size * i : batch_size * (i + 1)]
        if do_sample:
            insert_batch = insert_batch.sample(frac=args.sample_ratio, random_state=42)
        _, batch_histogram = make_histogram1d(
            insert_batch, indep, dep, resolution, bin_edges
        )
        batch_cum_sum = np.cumsum(batch_histogram)
        cum_sum += batch_cum_sum
        ys.append(cum_sum.copy())
    X_all = bin_edges[1:].reshape(-1, args.ndim_input)
    y_all = np.column_stack(ys)
    assert X_all.shape[0] == y_all.shape[0]
    return X_all, y_all


def save_traindata(name_recognizer):
    def inner(func):

        def wrapper(*pargs, **kwargs):
            args = pargs[0]
            # Load data if exists:
            data_name = args.data_name
            folder_name = data_name if data_name != "store_sales" else "tpc-ds"
            full_path = f"data/{folder_name}/traindata_{name_recognizer}_sr{args.sample_ratio}.npz"
            if os.path.exists(full_path):
                npzfile = np.load(full_path)
                X_train, y_train = npzfile["X"], npzfile["y"]
                try:
                    train_data_lens = npzfile["train_data_lens"]
                except KeyError:
                    return X_train, y_train
                else:
                    return X_train, y_train, train_data_lens

            outputs = func(*pargs, **kwargs)
            if len(outputs) == 2:
                X_train, y_train = outputs
                np.savez(full_path, X=X_train, y=y_train)
                return X_train, y_train
            elif len(outputs) == 3:
                X_train, y_train, train_data_lens = outputs
                # Save the train_data_lens as well
                np.savez(
                    full_path, X=X_train, y=y_train, train_data_lens=train_data_lens
                )
                return X_train, y_train, train_data_lens
            else:
                raise ValueError(f"Expected 2 or 3 outputs, but got {len(outputs)}")

        return wrapper

    return inner


@save_traindata("1D")
def prepare_training_data(args) -> tuple[np.ndarray, np.ndarray]:

    df = read_data(args)

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

    return X_train, y_train


@save_traindata("NHP")
def prepare_training_data_for_NHP(args):
    df = read_data(args)
    if args.sample_ratio < 1.0:
        df = df.sample(frac=args.sample_ratio, random_state=42)
    bin_edges, histogram = make_histogram1d(
        df, args.indeps[0], args.dep, args.resolutions[0]
    )
    X_train = bin_edges[1:].reshape(-1, 1)
    y_train = histogram.reshape(-1, 1)

    return X_train, y_train


@save_traindata("NHR")
def prepare_training_data_for_NHR(args):
    df = read_data(args)
    if args.sample_ratio < 1.0:
        df = df.sample(frac=args.sample_ratio, random_state=42)
    bin_edges, histogram = make_histogram1d(
        df, args.indeps[0], args.dep, args.resolutions[0]
    )

    X_train = []
    y_train = []
    train_data_lens = []
    range_percents = [0.05, 0.1, 0.15]

    for range_percent in range_percents:
        # Use sliding window to calculate the range sum
        query_range_size = int(len(histogram) * range_percent)
        train_data_len = len(histogram) - query_range_size
        range_sum = np.zeros(train_data_len)
        for i in range(train_data_len):
            range_sum[i] = np.sum(histogram[i : i + query_range_size])
        y_train.append(range_sum.reshape(-1, 1))

        X_train_ = bin_edges[:train_data_len]
        X_train_ = np.column_stack((X_train_, np.ones(train_data_len) * range_percent))
        X_train.append(X_train_)

        train_data_lens.append(train_data_len)

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    assert X_train.shape[0] == y_train.shape[0]
    train_data_lens = np.array(train_data_lens)

    return X_train, y_train, train_data_lens


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
