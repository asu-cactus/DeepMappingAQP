import numpy as np
from utils.parse_args import parse_args
from utils.data_utils import prepare_full_data, prepare_full_data_with_insertion
import pdb

# Set the random seed for reproducibility
np.random.seed(42)


def generate_1d_queries(args):
    X, y = prepare_full_data(args)

    X_max, X_min = X[-1][0], X[0][0]
    X_range = X_max - X_min
    query_percents = [0.1, 0.15, 0.2]
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

        indices = np.random.choice(
            len(X) - n_resolution - 1, args.nqueries, replace=True
        )
        start = X[indices]
        end = X[indices + n_resolution]
        label = y[indices + n_resolution] - y[indices]
        queries[str(query_percent)] = np.column_stack((start, end, label))

    np.savez(f"query/{args.data_name}_{args.task_type}_1D.npz", **queries)


def generate_1d_queries_nonzeros(args):
    X, y = prepare_full_data(args)

    X_max, X_min = X[-1][0], X[0][0]
    X_range = X_max - X_min

    query_percents = [0.05, 0.1, 0.15]
    # query_percents = [0.1]

    queries = {}
    for query_percent in query_percents:
        q_array = None

        skip_query_percent = False
        while q_array is None or len(q_array) < args.nqueries:
            # Random sample nqueries from X and y
            n_resolution = int((X_range * query_percent) / args.resolutions[0])
            if not n_resolution > 0:
                skip_query_percent = True
                break

            start_max_index = len(X) - n_resolution - 1
            if 10 * args.nqueries > start_max_index:
                sample_size = args.nqueries
            else:
                sample_size = 10 * args.nqueries
            try:
                indices = np.random.choice(start_max_index, sample_size, replace=False)
            except:
                pdb.set_trace()
            start = X[indices]
            end = X[indices + n_resolution]
            label = y[indices + n_resolution] - y[indices]
            q_batch = np.column_stack((start, end, label))
            # Filter out queries with zero label
            q_batch = q_batch[q_batch[:, 2] > 0]
            if q_array is None:
                q_array = q_batch
            else:
                q_array = np.vstack((q_array, q_batch))
        if skip_query_percent:
            continue
        queries[str(query_percent)] = q_array[: args.nqueries]

    np.savez(f"query/{args.data_name}_{args.task_type}_1D_nonzeros.npz", **queries)


def generate_1d_insert_queries_nonzero(args):
    X, ys = prepare_full_data_with_insertion(args)

    X_max, X_min = X[-1][0], X[0][0]
    X_range = X_max - X_min
    # y_max, y_min = y[-1][0], y[0][0]
    # y_range = y_max - y_min
    query_percents = [0.1]

    queries = {}
    for query_percent in query_percents:
        n_resolution = int((X_range * query_percent) / args.resolutions[0])
        if not n_resolution > 0:
            continue

        # The first column of ys is the original y values before insertion
        y_origin = ys[:, 0]

        for _, y in enumerate(ys.T):
            q_array = None
            while q_array is None or len(q_array) < args.nqueries:
                # Random sample nqueries from X and y

                start_max_index = len(X) - n_resolution - 1
                sample_size = 10 * args.nqueries
                indices = np.random.choice(start_max_index, sample_size, replace=False)
                start = X[indices]
                end = X[indices + n_resolution]
                label = y[indices + n_resolution] - y[indices]
                label_origin = y_origin[indices + n_resolution] - y_origin[indices]

                q_batch = np.column_stack((start, end, label, label_origin))
                # Filter out queries with zero label
                q_batch = q_batch[q_batch[:, 2] != 0]

                if q_array is None:
                    q_array = q_batch
                else:
                    q_array = np.vstack((q_array, q_batch))

            query_percent_str = str(query_percent)
            if query_percent_str not in queries:
                queries[query_percent_str] = []

            queries[query_percent_str].append(q_array[: args.nqueries])

    np.savez(f"query/{args.data_name}_insert_1D_nonzeros.npz", **queries)


def generate_1d_same_range_insert_queries_nonzero(args):
    raise NotImplementedError


def get_X_dimensions(X, resolutions):
    X_maxs, X_mins = X[-1], X[0]
    X_range_dim1 = X_maxs[0] - X_mins[0]
    X_range_dim2 = X_maxs[1] - X_mins[1]

    dim1_n_resol = int(X_range_dim1 / resolutions[0]) + 1
    dim2_n_resol = int(X_range_dim2 / resolutions[1]) + 1
    assert dim1_n_resol * dim2_n_resol == len(X)
    return X_range_dim1, X_range_dim2, dim1_n_resol, dim2_n_resol


def generate_2d_queries(args):

    X, y = prepare_full_data(args)

    X_range_dim1, X_range_dim2, dim1_n_resol, dim2_n_resol = get_X_dimensions(
        X, args.resolutions
    )

    queries = {}
    for query_percent in [0.15, 0.2, 0.25, 0.3]:
        # Random sample nqueries from X and y
        n_resol_dim1 = int((X_range_dim1 * query_percent) / args.resolutions[0])
        n_resol_dim2 = int((X_range_dim2 * query_percent) / args.resolutions[1])

        start_indices_dim1 = np.random.choice(
            dim1_n_resol - n_resol_dim1 - 1, args.nqueries, replace=True
        )
        start_indices_dim2 = np.random.choice(
            dim2_n_resol - n_resol_dim2 - 1, args.nqueries, replace=True
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
    args = parse_args()
    if args.ndim_input == 1:
        # generate_1d_queries(args)
        generate_1d_queries_nonzeros(args)
        if args.run_inserts:
            generate_1d_insert_queries_nonzero(args)
    else:
        generate_2d_queries(args)
