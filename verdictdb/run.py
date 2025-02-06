import random
import pymysql
import pyverdict
import pdb
from time import time
from memory_profiler import memory_usage

import pandas as pd
import numpy as np


def add_data():
    start = time()
    df = pd.read_csv(f"data/{data_name}/dataset.csv", usecols=[dep, indep])

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
    print("Data added to database in", time() - start, "seconds")


def create_verdict_conn():

    verdict_conn = pyverdict.mysql(
        host="localhost", user="root", password="", port=3306
    )
    return verdict_conn


def create_scramble_table(verdict_conn):
    # create scramble table
    start = time()
    verdict_conn.sql(f"DROP ALL SCRAMBLE {data_name}.{dep};")
    verdict_conn.sql(
        f"CREATE SCRAMBLE {data_name}.{dep}_scrambled from {data_name}.{dep}"
    )
    print(f"Scramble table created in {time() - start} seconds")


def query(verdict_conn, queries):

    # run query

    total_rel_error = 0.0
    for query in queries:
        X, y = query[:2], query[2]
        df = verdict_conn.sql(
            f"SELECT SUM({dep}) FROM {data_name}.{dep} WHERE {indep} BETWEEN {X[0]} AND {X[1]}"
        )
        y_hat = df.iloc[0, 0] if df.iloc[0, 0] is not None else 0
        relative_error = abs(y - y_hat) / (y + 1e-6)
        total_rel_error += relative_error
    print(f"Average relative error: {total_rel_error/nqueries}")
    # df = verdict_conn.sql(
    #     "SELECT SUM(taxi_out) "
    #     + "FROM flights.taxi_scrambled WHERE distance BETWEEN 800 AND 1000"
    # )

    # print(f"Result:\n{df.iloc[0][0]}")
    # print("Query executed in", time() - start, "seconds")

    # pdb.set_trace()


if __name__ == "__main__":

    task_type = "count"

    data_name = "ccpp"
    indep = "RH"
    dep = "PE"

    # data_name = "pm25"
    # indep = "PRES"
    # dep = "pm25"
    nqueries = 10
    npzfile = np.load(f"query/{data_name}_{indep}_{task_type}.npz")
    verdict_conn = create_verdict_conn()

    add_data()

    # Create scramble table
    mem = max(memory_usage((create_scramble_table, (verdict_conn,))))
    print(f"Maximum memory used for creating scramble table: {mem} MiB")

    # Start querying using scramble table
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][:nqueries]
        start = time()
        mem = max(memory_usage((query, (verdict_conn, queries))))
        print(f"Query percent: {query_percent} executed in {time() - start} seconds")
        print(f"Query percent: {query_percent}, Maximum memory used for : {mem} MiB")
