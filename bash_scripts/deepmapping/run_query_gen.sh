#!/bin/bash

python run_query_gen.py --data_name pm25 --run_inserts
python run_query_gen.py --data_name ccpp --run_inserts
python run_query_gen.py --data_name flights --run_inserts
python run_query_gen.py --data_name store_sales --run_inserts
python run_query_gen.py --data_name part
python run_query_gen.py --data_name lineitem