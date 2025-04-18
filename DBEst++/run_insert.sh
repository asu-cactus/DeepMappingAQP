#!/bin/bash

python3 query.py --data_name pm25 --units 440 --run_inserts
python3 query.py --data_name ccpp --units 430 --run_inserts
python3 query.py --data_name flights --units 500 --run_inserts
python3 query.py --data_name store_sales --units 980 --run_inserts