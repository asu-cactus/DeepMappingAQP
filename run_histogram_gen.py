from utils.parse_args import parse_args
from utils.data_utils import make_histogram1d, read_data
import pandas as pd
import numpy as np
import pdb


def prepare_histogram1d(args):

    df = read_data(args.data_name, args.task_type)
    if args.sample_ratio < 1.0:
        df = df.sample(frac=args.sample_ratio, random_state=42)
    bin_edges, histogram = make_histogram1d(
        df, args.indeps[0], args.dep, args.resolutions[0]
    )

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    # Combine bin_centers and histogram into a single array and convert to a dataframe
    hist_df = pd.DataFrame(
        np.column_stack((bin_centers, histogram)), columns=[args.indeps[0], args.dep]
    )

    hist_df.to_csv(f"data/{args.data_name}/histogram1d.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    if args.ndim_input == 1:
        prepare_histogram1d(args)
