import pymysql
import pyverdict


def cleanup_mysql():
    mysql_conn = pymysql.connect(
        host="localhost", port=3306, user="root", passwd="", autocommit=True
    )
    cur = mysql_conn.cursor()
    cur.execute("DROP SCHEMA IF EXISTS flights")
    cur.execute("DROP SCHEMA IF EXISTS ccpp")
    cur.execute("DROP SCHEMA IF EXISTS pm25")
    cur.execute("DROP SCHEMA IF EXISTS store_sales")
    cur.close()


def cleanup_scramble():
    verdict_conn = pyverdict.mysql(
        host="localhost", user="root", password="", port=3306
    )
    # scrambles = verdict_conn.sql("SHOW SCRAMBLES;")
    # print(scrambles)

    verdict_conn.sql("DROP ALL SCRAMBLE flights.TAXI_OUT;")
    verdict_conn.sql("DROP ALL SCRAMBLE ccpp.PE;")
    verdict_conn.sql("DROP ALL SCRAMBLE pm25.pm25;")
    verdict_conn.sql("DROP ALL SCRAMBLE store_sales.wholesale_cost;")


if __name__ == "__main__":
    cleanup_scramble()
    cleanup_mysql()
