#!/bin/bash
python verdictdb/data2mysql.py --data_name pm25
python verdictdb/data2mysql.py --data_name ccpp
python verdictdb/data2mysql.py --data_name flights
python verdictdb/data2mysql.py --data_name store_sales