from utils.parse_args import parse_args

from utils.data_utils import prepare_training_data
import numpy as np
from time import perf_counter
import pdb


def test(
    args,
    synopsis: np.array,
    query_path: str,
    X_min: float,
    total_sum: float,
    **kwargs,
):

    # Load queries
    npzfile = np.load(query_path)
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][: args.nqueries]

        # The first column is the start of the query range and the second column is the end
        start = perf_counter()
        total_rel_error = 0.0
        total_error_II = 0.0
        for query in queries:
            X, y = query[:2], query[2]
            indices = ((X - X_min) / args.resolutions[0]).round().astype(int)
            y_out = synopsis[indices]
            y_hat = (y_out[1] - y_out[0]) / args.sample_ratio

            rel_error = np.absolute(y_hat - y) / (y + 1e-6)
            error_II = np.absolute(y_hat - y) / total_sum

            total_rel_error += rel_error.item()
            total_error_II += error_II.item()

        avg_rel_error = total_rel_error / len(queries)
        avg_error_II = total_error_II / len(queries)
        avg_time = (perf_counter() - start) / len(queries)

        print(f"{query_percent} query:  executed in {avg_time} seconds on average.")
        print(f"Avg rel error: {avg_rel_error:.4f}, Avg error II: {avg_error_II:.4f}")


def main(args):

    # Prepare training data

    X, synopsis = prepare_training_data(args)
    print(f"synopsis size: {4 * synopsis.size / 1024:.2f} KB")
    X_min = X[0][0] if args.ndim_input == 1 else X[0]
    total_sum = synopsis[-1][0]

    # Run test
    query_path = f"query/{args.data_name}_{args.task_type}_{args.ndim_input}D.npz"

    test(
        args,
        synopsis,
        query_path,
        X_min,
        total_sum,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
