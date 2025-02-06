import numpy as np
from parse_args import parse_args
from data_utils import read_data, prepare_training_data
import pdb

# Set the random seed for reproducibility
np.random.seed(42)


def generate_queries(nqueries):
    args = parse_args()
    df = read_data(args.data_name)
    X, y = prepare_training_data(
        df,
        args.indep,
        args.dep,
        args.resolution,
        args.output_scale,
        do_standardize=False,
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
        n_resolution = int((X_range * query_percent) / args.resolution)
        if not n_resolution > 0:
            continue

        indices = np.random.choice(len(X) - n_resolution - 1, nqueries, replace=True)
        start = X[indices]
        end = X[indices + n_resolution]
        label = y[indices + n_resolution] - y[indices]
        queries[str(query_percent)] = np.column_stack((start, end, label))

    np.savez(f"query/{args.data_name}_{args.indep}_{args.task_type}.npz", **queries)


if __name__ == "__main__":
    generate_queries(nqueries=1000)
