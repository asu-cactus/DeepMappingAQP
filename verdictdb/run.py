import pyverdict
import pandas as pd
import numpy as np

import pdb
import os
import argparse
from time import perf_counter

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VerdictDB for AQP")
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument("--do_insert", action="store_true", help="Do insertion")
    parser.add_argument("--task_type", type=str, default="sum", help="Task type")
    parser.add_argument("--nqueries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--from_histogram", action="store_true", help="From sypnosis")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="Sample ratio")
    parser.add_argument("--ndim_input", type=int, default=1, help="Input dimension")
    args = parser.parse_args()

    if args.from_histogram:
        args.sample_ratio = 1.0
    print(args)
    return args


def create_verdict_conn():
    start = perf_counter()
    verdict_conn = pyverdict.mysql(
        host="localhost", user="root", password="", port=3306
    )
    print(f"VerdictDB connection created in {perf_counter() - start:.4f} seconds")
    return verdict_conn


def get_size(scramble_table_name):
    df = verdict_conn.sql(f"SELECT COUNT(*) FROM {scramble_table_name};")
    nrows = df.iloc[0, 0]
    ncols = 2  # 4 columns if extra columns are considered
    bytes_per_value = 4
    size = nrows * ncols * bytes_per_value / 1024
    return size


def create_scramble_table(table_name):
    scramble_table_name = f"{table_name}_scrambled"
    # create scramble table
    start = perf_counter()
    df = verdict_conn.sql(f"SELECT * FROM {table_name} LIMIT 10;")
    print(df.head())
    try:
        verdict_conn.sql(f"DROP SCRAMBLE {scramble_table_name} ON {table_name};")
        # verdict_conn.sql(f"SELECT COUNT(*) FROM {scramble_table_name};")
    except:
        print(f"Scramble {scramble_table_name} table does not exist")

    verdict_conn.sql(
        f"CREATE SCRAMBLE {scramble_table_name} FROM {table_name} RATIO {args.sample_ratio} WHERE nth_insert=0;"
    )
    print(f"Scramble table created in {perf_counter() - start:.4f} seconds")

    # Get number of rows in scramble table
    original_size = get_size(scramble_table_name)

    return (scramble_table_name, original_size)


def save_results(results, save_path):
    if os.path.exists(save_path):
        result_df = pd.read_csv(save_path)
        result_df = pd.concat([result_df, pd.DataFrame(results)], ignore_index=True)
    else:
        result_df = pd.DataFrame(results)

    result_df.to_csv(save_path, index=False)


def query(args, queries, scramble_table_name, query_percent, ith_insert):
    # run query
    start = perf_counter()
    total_rel_error = 0.0
    for query in queries:
        X, y = query[:2], query[2]
        df = verdict_conn.sql(
            f"SELECT SUM({dep}) FROM {scramble_table_name} WHERE {indep} BETWEEN {X[0]} AND {X[1]}"
        )

        if not isinstance(df.iloc[0, 0], float):
            print(f"Irregular result: {df.iloc[0, 0]}")

        y_hat = df.iloc[0, 0] if isinstance(df.iloc[0, 0], float) else 0

        # Change #3
        y_hat /= args.sample_ratio

        relative_error = abs(y - y_hat) / (y + EPS)
        print(f"y: {y}, y_hat: {y_hat}, relative error: {relative_error}")
        total_rel_error += relative_error
    avg_rel_error = total_rel_error / args.nqueries
    avg_query_time = (perf_counter() - start) / args.nqueries

    query_percent = int(float(query_percent) * 100)
    print(f"{query_percent}%-{ith_insert}th insert, avg_rel_error: {avg_rel_error:.4f}")
    print(f"Avg execute time {avg_query_time:.4f} seconds")

    return avg_rel_error, avg_query_time


def query_original(scramble_table_name, original_size, save_path):
    results = []
    npzfile = np.load(f"query/{args.data_name}_{args.task_type}_1D_nonzeros.npz")
    for query_percent in npzfile.keys():

        queries = npzfile[query_percent][: args.nqueries]
        avg_rel_error, avg_query_time = query(
            args, queries, scramble_table_name, query_percent, 0
        )

        results.append(
            {
                "query_percent": query_percent,
                "original_size(KB)": round(original_size, 2),
                "size(KB)": round(original_size, 2),
                "nth_insert": 0,
                "avg_rel_error": round(avg_rel_error, 4),
                "avg_query_time": round(avg_query_time, 6),
            }
        )
    save_results(results, save_path)


def query_after_insertion(args, scramble_table_name, original_size, save_path):
    results = []
    # Run queries
    query_path = f"query/{args.data_name}_insert_{args.ndim_input}D_nonzeros.npz"
    npzfile = np.load(query_path, allow_pickle=True)

    for query_percent, query_group in npzfile.items():
        for i, queries in enumerate(query_group):
            # if i == 0:
            #     # The 0th are queries of the original data
            #     continue
            queries = queries[: args.nqueries]

            verdict_conn.sql(
                f"APPEND SCRAMBLE {scramble_table_name} WHERE nth_insert={i}"
            )
            avg_rel_error, avg_query_time = query(
                args, queries, scramble_table_name, query_percent, i
            )
            curr_size = get_size(scramble_table_name)

            results.append(
                {
                    "query_percent": query_percent,
                    "original_size(KB)": round(original_size, 2),
                    "size(KB)": round(curr_size, 2),
                    "nth_insert": i,
                    "avg_rel_error": round(avg_rel_error, 4),
                    "avg_query_time": round(avg_query_time, 6),
                }
            )

    save_results(results, save_path)


if __name__ == "__main__":

    args = parse_args()

    task_type = args.task_type

    if args.data_name == "store_sales":
        indep = "list_price"
        dep = "wholesale_cost"
    elif args.data_name == "flights":
        indep = "DISTANCE"
        dep = "TAXI_OUT"
    elif args.data_name == "pm25":
        indep = "PRES"
        dep = "pm25"
    elif args.data_name == "ccpp":
        indep = "RH"
        dep = "PE"
    else:
        raise ValueError(f"No support for {args.data_name} for 1D input")

    table_name = f"{args.data_name}.{dep}"

    insert_str = ""
    if args.do_insert:
        if args.uniform_update:
            insert_str = "_insert"
        else:
            insert_str = "_insert_hotregion"
    save_path = f"results/{args.data_name}_verdictdb{insert_str}.csv"
    # Create scramble table
    verdict_conn = create_verdict_conn()

    scramble_table_name, original_size = create_scramble_table(table_name)
    query_original(scramble_table_name, original_size, save_path)
    if args.do_insert:
        query_after_insertion(args, scramble_table_name, original_size, save_path)
