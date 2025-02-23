import pymysql
import pyverdict
import pandas as pd
import numpy as np

import pdb
import argparse
from time import perf_counter
from memory_profiler import memory_usage

EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VerdictDB for AQP")
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument("--task_type", type=str, default="sum", help="Task type")
    parser.add_argument("--nqueries", type=int, default=10, help="Number of queries")
    parser.add_argument("--from_histogram", action="store_true", help="From sypnosis")
    parser.add_argument("--sample_ratio", type=float, default=0.1, help="Sample ratio")
    parser.add_argument("--ndim_input", type=int, default=1, help="Input dimension")

    args = parser.parse_args()

    if args.from_histogram:
        args.sample_ratio = 1.0
    print(args)
    return args


def add_data(data_name):
    start = perf_counter()
    foldername = data_name if data_name != "store_sales" else "tpc-ds"
    filename = (
        f"histogram{args.ndim_input}d.csv"
        if args.from_histogram
        else f"dataset_{args.task_type}.csv"
    )
    df = pd.read_csv(f"data/{foldername}/{filename}", usecols=[dep, indep])

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
    verdict_conn.sql(
        f"CREATE SCRAMBLE {data_name}.{dep}_scrambled FROM {data_name}.{dep} RATIO {args.sample_ratio};"
    )
    print(f"Scramble table created in {perf_counter() - start:.4f} seconds")


def query(args, verdict_conn, queries):

    # run query

    total_rel_error = 0.0
    for query in queries:
        X, y = query[:2], query[2]
        df = verdict_conn.sql(
            f"SELECT SUM({dep}) FROM {args.data_name}.{dep} WHERE {indep} BETWEEN {X[0]} AND {X[1]}"
        )
        if not isinstance(df.iloc[0, 0], float):
            print(f"Irregular result:\n{df.iloc[0, 0]}")
        y_hat = df.iloc[0, 0] if isinstance(df.iloc[0, 0], float) else 0
        relative_error = abs(y - y_hat) / (y + EPS)
        print(f"Relative error: {relative_error}")
        total_rel_error += relative_error
    print(f"Average relative error: {total_rel_error/args.nqueries}")
    # df = verdict_conn.sql(
    #     "SELECT SUM(taxi_out) "
    #     + "FROM flights.taxi_scrambled WHERE distance BETWEEN 800 AND 1000"
    # )

    # print(f"Result:\n{df.iloc[0][0]}")
    # print("Query executed in", time() - start, "seconds")


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

    npzfile = np.load(f"query/{args.data_name}_{args.task_type}_1D.npz")
    verdict_conn = create_verdict_conn()

    add_data(args.data_name)

    # Create scramble table
    mem = max(
        memory_usage(
            (
                create_scramble_table,
                (args, verdict_conn),
            )
        )
    )
    print(f"Maximum memory used for creating scramble table: {mem} MiB")

    # Start querying using scramble table
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][: args.nqueries]
        start = perf_counter()
        mem = max(
            memory_usage(
                (
                    query,
                    (args, verdict_conn, queries),
                )
            )
        )
        print(
            f"Query percent: {query_percent} executed in {perf_counter() - start} seconds"
        )
        print(f"Query percent: {query_percent}, Maximum memory used for : {mem} MiB")
