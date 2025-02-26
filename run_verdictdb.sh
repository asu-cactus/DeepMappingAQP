#!/bin/bash

python verdictdb/run.py --data_name store_sales --sample_ratio 0.010
python verdictdb/run.py --data_name store_sales --sample_ratio 0.009 
python verdictdb/run.py --data_name store_sales --sample_ratio 0.008
python verdictdb/run.py --data_name store_sales --sample_ratio 0.007
python verdictdb/run.py --data_name store_sales --sample_ratio 0.006
python verdictdb/run.py --data_name store_sales --sample_ratio 0.005
python verdictdb/run.py --data_name store_sales --sample_ratio 0.004
python verdictdb/run.py --data_name store_sales --sample_ratio 0.003