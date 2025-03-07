import pandas as pd
import json
import pdb
from utils.data_utils import read_insertion_data, read_data, prepare_full_data
from utils.update_1d import get_update_ranges
from utils.parse_args import parse_args


def get_update_ranges_1d(args):
    indep = args.indeps[0]
    df = read_data(args)
    X_min = df[indep].min()
    X_max = df[indep].max()

    ranges = get_update_ranges(X_min, X_max, args)
    return ranges


def save_ranges_as_json(ranges, args):
    json_obj = {
        f"range{i}": {"start": r.start, "end": r.end} for i, r in enumerate(ranges)
    }
    folder_name = args.data_name if args.data_name != "store_sales" else "tpc-ds"
    with open(f"data/update_data/{folder_name}/ranges.json", "w") as f:
        json.dump(json_obj, f)


def filter_insertion_data(args):
    df_origin = read_insertion_data(args)
    ranges = get_update_ranges_1d(args)
    save_ranges_as_json(ranges, args)

    # Only keep df[indep] that are within the ranges
    indep = args.indeps[0]
    update_dfs = []
    for r in ranges:
        update_dfs.append(
            df_origin[(df_origin[indep] >= r.start) & (df_origin[indep] < r.end)]
        )
    # Concatenate all the update_dfs
    df = pd.concat(update_dfs)

    # Make the length of df to be n_insert
    replace = len(df) < args.n_insert
    try:
        df = df.sample(args.n_insert, replace=replace)
    except:
        # It is possible that len(df) == 0
        pdb.set_trace()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the filtered insertion data
    folder_name = args.data_name if args.data_name != "store_sales" else "tpc-ds"
    df.to_csv(f"data/update_data/{folder_name}/insert.csv", index=False)
    return df


if __name__ == "__main__":
    args = parse_args()
    filter_insertion_data(args)
