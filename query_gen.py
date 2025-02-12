import numpy as np
from parse_args import parse_args
from utils.data_utils import prepare_training_data
import pdb

# Set the random seed for reproducibility
np.random.seed(42)


def generate_1d_queries(nqueries):
    args = parse_args()
    X, y = prepare_training_data(
        args.data_name,
        args.task_type,
        args.indeps,
        args.dep,
        args.resolutions,
        args.ndim_input,
    )

    X_max, X_min = X[-1][0], X[0][0]
    X_range = X_max - X_min
    query_percents = [0.001, 0.01, 0.05, 0.1]
    # query_ranges = [
    #     (X_range * query_percent) // args.resolution * args.resolution
    #     for query_percent in query_percents
    # ]

    queries = {}
    for query_percent in query_percents:
        # Random sample nqueries from X and y
        n_resolution = int((X_range * query_percent) / args.resolutions[0])
        if not n_resolution > 0:
            continue

        indices = np.random.choice(len(X) - n_resolution - 1, nqueries, replace=True)
        start = X[indices]
        end = X[indices + n_resolution]
        label = y[indices + n_resolution] - y[indices]
        queries[str(query_percent)] = np.column_stack((start, end, label))

    np.savez(f"query/{args.data_name}_{args.task_type}_1D.npz", **queries)


def get_X_dimensions(X, resolutions):
    X_maxs, X_mins = X[-1], X[0]
    X_range_dim1 = X_maxs[0] - X_mins[0]
    X_range_dim2 = X_maxs[1] - X_mins[1]

    dim1_n_resol = int(X_range_dim1 / resolutions[0]) + 1
    dim2_n_resol = int(X_range_dim2 / resolutions[1]) + 1
    assert dim1_n_resol * dim2_n_resol == len(X)
    return X_range_dim1, X_range_dim2, dim1_n_resol, dim2_n_resol


def generate_2d_queries(nqueries):
    args = parse_args()

    X, y = prepare_training_data(
        args.data_name,
        args.task_type,
        args.indeps,
        args.dep,
        args.resolutions,
        args.ndim_input,
    )

    X_range_dim1, X_range_dim2, dim1_n_resol, dim2_n_resol = get_X_dimensions(
        X, args.resolutions
    )

    queries = {}
    for query_percent in [0.05, 0.1, 0.15, 0.2]:
        # Random sample nqueries from X and y
        n_resol_dim1 = int((X_range_dim1 * query_percent) / args.resolutions[0])
        n_resol_dim2 = int((X_range_dim2 * query_percent) / args.resolutions[1])

        start_indices_dim1 = np.random.choice(
            dim1_n_resol - n_resol_dim1 - 1, nqueries, replace=True
        )
        start_indices_dim2 = np.random.choice(
            dim2_n_resol - n_resol_dim2 - 1, nqueries, replace=True
        )
        end_indices_dim1 = start_indices_dim1 + n_resol_dim1
        end_indices_dim2 = start_indices_dim2 + n_resol_dim2

        lower_left_index = start_indices_dim1 * dim2_n_resol + start_indices_dim2
        upper_right_index = end_indices_dim1 * dim2_n_resol + end_indices_dim2
        lower_right_index = end_indices_dim1 * dim2_n_resol + start_indices_dim2
        upper_left_index = start_indices_dim1 * dim2_n_resol + end_indices_dim2

        start = X[lower_left_index]
        end = X[upper_right_index]
        label = (
            y[upper_right_index]
            - y[lower_right_index]
            - y[upper_left_index]
            + y[lower_left_index]
        )

        queries[str(query_percent)] = np.column_stack((start, end, label))

    np.savez(f"query/{args.data_name}_{args.task_type}_2D.npz", **queries)


if __name__ == "__main__":
    generate_1d_queries(nqueries=10000)
    # generate_2d_queries(nqueries=10000)
