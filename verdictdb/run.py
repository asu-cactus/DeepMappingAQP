import pymysql
import pyverdict
import pandas as pd
import numpy as np

import pdb
import os
import argparse
from time import perf_counter

# from memory_profiler import memory_usage

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VerdictDB for AQP")
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument("--task_type", type=str, default="sum", help="Task type")
    parser.add_argument("--nqueries", type=int, default=100, help="Number of queries")
    parser.add_argument("--from_histogram", action="store_true", help="From sypnosis")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Sample ratio")
    parser.add_argument("--ndim_input", type=int, default=1, help="Input dimension")

    args = parser.parse_args()

    if args.from_histogram:
        args.sample_ratio = 1.0
    print(args)
    return args


def add_data(args):
    data_name = args.data_name
    start = perf_counter()
    foldername = data_name if data_name != "store_sales" else "tpc-ds"
    filename = (
        f"histogram{args.ndim_input}d.csv"
        if args.from_histogram
        else f"dataset_{args.task_type}.csv"
    )
    df = pd.read_csv(f"data/{foldername}/{filename}", usecols=[dep, indep])

    # Change #1
    df = df.sample(frac=args.sample_ratio, random_state=42)

    mysql_conn = pymysql.connect(
        host="localhost", port=3306, user="root", passwd="", autocommit=True
    )
    cur = mysql_conn.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {data_name}")
    cur.execute(f"CREATE SCHEMA {data_name}")
    cur.execute(f"CREATE TABLE {data_name}.{dep} ({dep} double, {indep} double)")

    cur.executemany(
        f"INSERT INTO {data_name}.{dep} VALUES (%s, %s)", df.values.tolist()
    )

    cur.close()
    print(f"Data added to database in {perf_counter() - start:.4f} seconds")

    size_in_KB = len(df) * 2 * 4 / 1024
    return size_in_KB


def create_verdict_conn():

    verdict_conn = pyverdict.mysql(
        host="localhost", user="root", password="", port=3306
    )
    return verdict_conn


def create_scramble_table(args, verdict_conn):
    data_name = args.data_name
    # create scramble table
    start = perf_counter()
    verdict_conn.sql(f"DROP ALL SCRAMBLE {data_name}.{dep};")

    # Change #2
    # ratio_str = f"RATIO {args.sample_ratio}"
    ratio_str = f"RATIO 1.0"

    verdict_conn.sql(
        f"CREATE SCRAMBLE {data_name}.{dep}_scrambled FROM {data_name}.{dep} {ratio_str};"
    )
    print(f"Scramble table created in {perf_counter() - start:.4f} seconds")


def query(args, verdict_conn, queries):

    # run query
    start = perf_counter()
    total_rel_error = 0.0
    for query in queries:
        X, y = query[:2], query[2]
        df = verdict_conn.sql(
            f"SELECT SUM({dep}) FROM {args.data_name}.{dep} WHERE {indep} BETWEEN {X[0]} AND {X[1]}"
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
    print(
        f"Query percent: {query_percent}, average relative error: {avg_rel_error:.4f}"
    )
    print(f"Avg execute time {avg_query_time:.4f} seconds")

    return avg_rel_error, avg_query_time


if __name__ == "__main__":

    args = parse_args()

    task_type = "sum"

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

    npzfile = np.load(f"query/{args.data_name}_{args.task_type}_1D_nonzeros.npz")
    verdict_conn = create_verdict_conn()

    size_in_KB = add_data(args)

    # Create scramble table
    create_scramble_table(args, verdict_conn)

    # Start querying using scramble table

    # Load dataframe from save_path if exists
    save_path = f"results/{args.data_name}_verdictdb.csv"
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(
            columns=["size(KB)", "query_percent", "avg_rel_error", "avg_query_time"]
        )

    # Collect results and append to dataframe
    results = []
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][: args.nqueries]
        avg_rel_error, avg_query_time = query(args, verdict_conn, queries)
        avg_rel_error = round(avg_rel_error, 4)
        avg_query_time = round(avg_query_time, 4)
        results.append([size_in_KB, query_percent, avg_rel_error, avg_query_time])

    df = pd.concat([df, pd.DataFrame(results, columns=df.columns)], ignore_index=True)
    df.to_csv(save_path, index=False)
