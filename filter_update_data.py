import pandas as pd
import json
import pdb
from utils.data_utils import read_insertion_data, read_data
from utils.update_1d import get_update_ranges, Range
from utils.parse_args import parse_args


def filter_insertion_data(args):
    if args.uniform_update:
        filter_insertion_by_size(args)
    else:
        filter_insertion_by_ranges_and_size(args)


def filter_insertion_by_size(args):
    df_origin = read_insertion_data(args, "insert_origin.csv")
    replace = len(df_origin) < args.n_insert

    df = df_origin.sample(args.n_insert, replace=replace)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    folder_name = args.data_name if args.data_name != "store_sales" else "tpc-ds"
    df.to_csv(
        f"data/update_data/{folder_name}/insert_filtered_by_size.csv", index=False
    )

    indep = args.indeps[0]
    X_min = int(df_origin[indep].min())
    X_max = int(df_origin[indep].max())
    ranges = [Range(X_min, X_max) for _ in range(5)]
    ranges_for_nruns = [
        ranges
        for _ in range(
            1
            if args.no_retrain
            else int(args.n_insert_batch / args.retrain_every_n_insert)
        )
    ]
    save_ranges_as_json(ranges_for_nruns, args)


def get_update_ranges_1d(args):
    indep = args.indeps[0]
    df = read_data(args)
    X_min = df[indep].min()
    X_max = df[indep].max()

    ranges = get_update_ranges(X_min, X_max, args)
    return ranges


def save_ranges_as_json(ranges_for_nruns, args):
    json_obj = {}
    for i, ranges in enumerate(ranges_for_nruns):
        json_obj[f"run{i}"] = [{"start": r.start, "end": r.end} for r in ranges]

    folder_name = args.data_name if args.data_name != "store_sales" else "tpc-ds"
    filename = "ranges_by_size.json" if args.uniform_update else "ranges.json"
    with open(f"data/update_data/{folder_name}/{filename}", "w") as f:
        json.dump(json_obj, f)


def filter_insertion_by_ranges_and_size(args):
    df_origin = read_insertion_data(args, "insert_origin.csv")

    if args.no_retrain:
        n_runs = 1
    else:
        assert args.n_insert_batch % args.retrain_every_n_insert == 0
        n_runs = int(args.n_insert_batch / args.retrain_every_n_insert)

    n_insert_per_run = args.n_insert // n_runs

    dfs = []
    ranges_for_nruns = []
    for _ in range(n_runs):
        ranges = get_update_ranges_1d(args)
        ranges_for_nruns.append(ranges)

        # Only keep df[indep] that are within the ranges
        indep = args.indeps[0]
        update_dfs = []
        for r in ranges:
            update_dfs.append(
                df_origin[(df_origin[indep] >= r.start) & (df_origin[indep] < r.end)]
            )
        # Concatenate all the update_dfs
        df = pd.concat(update_dfs)

        # Make the length of df to be n_insert_per_run

        replace = len(df) < n_insert_per_run
        try:
            df = df.sample(n_insert_per_run, replace=replace)
        except:
            # It is possible that len(df) == 0
            pdb.set_trace()
        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        dfs.append(df)

    save_ranges_as_json(ranges_for_nruns, args)
    # Concatenate all the dfs and save it
    df = pd.concat(dfs)
    folder_name = args.data_name if args.data_name != "store_sales" else "tpc-ds"
    df.to_csv(
        f"data/update_data/{folder_name}/insert_filtered_by_range.csv", index=False
    )


if __name__ == "__main__":
    args = parse_args()
    filter_insertion_data(args)
