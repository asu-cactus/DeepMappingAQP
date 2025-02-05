import numpy as np
from parse_args import parse_args
from data_utils import read_data, prepare_training_data
import pdb

# Set the random seed for reproducibility
np.random.seed(42)


def generate_queries(nqueries):
    args = parse_args()
    df = read_data(args.data_name)
    indep, dep = args.indep, args.dep
    df = df[[indep, dep]]
    df = df.dropna()
    df = df.sort_values(by=indep)
    df[dep] = df[dep].cumsum()

    X_min = df.iloc[0][indep]
    X_max = df.iloc[-1][indep]
    print(f"X_min: {X_min}, X_max: {X_max}")

    query_percents = [0.001, 0.01, 0.05, 0.1]
    queries = {}
    for query_percent in query_percents:
        query_range = (X_max - X_min) * query_percent
        query_start = np.random.uniform(X_min, X_max - query_range, nqueries)
        query_start = query_start
        query_end = query_start + query_range
        # Compute the range sum as the answer
        start_pos = df[indep].searchsorted(query_start, side="right") - 1
        end_pos = df[indep].searchsorted(query_end, side="right")
        answer = df.loc[end_pos, dep].values - df.loc[start_pos, dep].values
        queries[str(query_percent)] = np.column_stack((query_start, query_end, answer))
    np.savez(f"query/{args.data_name}_{indep}_sum.npz", **queries)


def generate_queries_v2(nqueries):
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

    np.savez(f"query/{args.data_name}_{args.indep}_sum.npz", **queries)


if __name__ == "__main__":
    generate_queries_v2(nqueries=1000)
