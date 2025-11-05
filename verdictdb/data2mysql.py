import pymysql
import argparse
import pandas as pd
import numpy as np
from time import perf_counter
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add data to MySQL")
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument("--task_type", type=str, default="sum", help="Task type")
    parser.add_argument("--ndim_input", type=int, default=1, help="Input dimension")
    parser.add_argument("--from_histogram", action="store_true", help="From sypnosis")
    parser.add_argument("--uniform_update", action="store_true", help="Uniform update")

    args = parser.parse_args()
    print(args)
    return args


def create_mysql_conn():
    mysql_conn = pymysql.connect(
        host="localhost", port=3306, user="root", passwd="", autocommit=True
    )
    return mysql_conn


# def get_data_sample(args, data_name, dep, indep):
#     # Change #1
#     foldername = data_name if data_name != "store_sales" else "tpc-ds"
#     filename = (
#         f"histogram{args.ndim_input}d.csv"
#         if args.from_histogram
#         else f"dataset_{args.task_type}.csv"
#     )
#     if args.save_mem:
#         # Make sure that data is already randomly shuffled
#         if not args.from_histogram:
#             command = f"wc -l data/{foldername}/{filename}"
#             result = subprocess.run(command, capture_output=True, text=True, shell=True)

#             if result.returncode == 0:
#                 output = result.stdout
#                 # Get the number of rows in the file, first row is the header
#                 n_all_rows = int(output.split()[0]) - 1
#             else:
#                 error = result.stderr
#                 raise Exception(f"Shell command error: {error}")
#             nrows = int(n_all_rows * args.sample_ratio)
#         else:
#             nrows = None

#         df = pd.read_csv(
#             f"data/{foldername}/{filename}", usecols=[dep, indep], nrows=nrows
#         )[[dep, indep]].dropna()

#     else:
#         df = pd.read_csv(f"data/{foldername}/{filename}", usecols=[dep, indep])[
#             [dep, indep]
#         ].dropna()
#         df = df.sample(frac=args.sample_ratio, random_state=42)

#     return df


def add_original_data(args, cur, dep, indep, table_name):
    data_name = args.data_name
    foldername = data_name if data_name != "store_sales" else "tpc-ds"
    filename = (
        f"histogram{args.ndim_input}d.csv"
        if args.from_histogram
        else f"dataset_{args.task_type}.csv"
    )
    df = pd.read_csv(f"data/{foldername}/{filename}", usecols=[dep, indep])[
        [dep, indep]
    ]
    df["nth_insert"] = 0

    start = perf_counter()

    cur = mysql_conn.cursor()
    cur.execute(f"DROP SCHEMA IF EXISTS {data_name}")
    cur.execute(f"CREATE SCHEMA {data_name}")
    cur.execute(
        f"CREATE TABLE {table_name} ({dep} double, {indep} double, nth_insert int)"
    )

    cur.executemany(f"INSERT INTO {table_name} VALUES (%s, %s, %s)", df.values.tolist())

    print(f"Add original data to database in {perf_counter() - start:.2f} seconds")
    return table_name


def add_insertion_data(args, cur, dep, indep, table_name):
    data_name = args.data_name

    foldername = data_name if data_name != "store_sales" else "tpc-ds"

    filename = (
        "insert_filtered_by_size.csv"
        if args.uniform_update
        else "insert_filtered_by_range.csv"
    )
    df_insert = pd.read_csv(
        f"data/update_data/{foldername}/{filename}", header=0, usecols=[dep, indep]
    )[[dep, indep]]

    query_path = f"query/{args.data_name}_insert_{args.ndim_input}D_nonzeros.npz"
    npzfile = np.load(query_path, allow_pickle=True)

    start = perf_counter()
    for query_percent, query_group in npzfile.items():
        n_insert_batch = query_group.shape[0] - 1
        batch_size = int(len(df_insert) / n_insert_batch)
        for i, _ in enumerate(query_group):
            if i == 0:
                # The 0th are queries of the original data
                continue

            insert_batch = df_insert[batch_size * (i - 1) : batch_size * i].copy()
            insert_batch["nth_insert"] = i
            cur.executemany(
                f"INSERT INTO {table_name} VALUES (%s, %s, %s)",
                insert_batch.values.tolist(),
            )
        # Only insert once, no need to do it for all query_percent
        break

    print(f"Add insertion data to database in {perf_counter() - start:.2f} seconds")


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
    elif args.data_name == "part":
        indep = "partkey"
        dep = "retailprice"
    elif args.data_name == "lineitem":
        indep = "extendedprice"
        dep = "quantity"
    else:
        raise ValueError(f"No support for {args.data_name} for 1D input")

    table_name = f"{args.data_name}.{dep}"

    mysql_conn = create_mysql_conn()
    cur = mysql_conn.cursor()
    add_original_data(args, cur, dep, indep, table_name)
    add_insertion_data(args, cur, dep, indep, table_name)
    cur.close()
    mysql_conn.close()
