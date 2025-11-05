#!/bin/bash
python verdictdb/data2mysql.py --data_name pm25
python verdictdb/data2mysql.py --data_name ccpp
python verdictdb/data2mysql.py --data_name flights
python verdictdb/data2mysql.py --data_name store_sales
python verdictdb/data2mysql.py --data_name part
python verdictdb/data2mysql.py --data_name lineitem

# python verdictdb/data2mysql.py --data_name pm25 --uniform_update
# python verdictdb/data2mysql.py --data_name ccpp --uniform_update
# python verdictdb/data2mysql.py --data_name flights --uniform_update
# python verdictdb/data2mysql.py --data_name store_sales --uniform_update