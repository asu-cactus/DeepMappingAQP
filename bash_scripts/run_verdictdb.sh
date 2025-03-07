#!/bin/bash

python verdictdb/run.py --data_name store_sales --sample_ratio 0.010
python verdictdb/run.py --data_name store_sales --sample_ratio 0.009 
python verdictdb/run.py --data_name store_sales --sample_ratio 0.008
python verdictdb/run.py --data_name store_sales --sample_ratio 0.007
python verdictdb/run.py --data_name store_sales --sample_ratio 0.006
python verdictdb/run.py --data_name store_sales --sample_ratio 0.005
python verdictdb/run.py --data_name store_sales --sample_ratio 0.004
python verdictdb/run.py --data_name store_sales --sample_ratio 0.003

python verdictdb/run.py --data_name flights --sample_ratio 0.0030
python verdictdb/run.py --data_name flights --sample_ratio 0.0028
python verdictdb/run.py --data_name flights --sample_ratio 0.0025
python verdictdb/run.py --data_name flights --sample_ratio 0.0020
python verdictdb/run.py --data_name flights --sample_ratio 0.0018
python verdictdb/run.py --data_name flights --sample_ratio 0.0015

python verdictdb/run.py --data_name ccpp --sample_ratio 0.0035
python verdictdb/run.py --data_name ccpp --sample_ratio 0.0030
python verdictdb/run.py --data_name ccpp --sample_ratio 0.0025
python verdictdb/run.py --data_name ccpp --sample_ratio 0.0020
python verdictdb/run.py --data_name ccpp --sample_ratio 0.0015
python verdictdb/run.py --data_name ccpp --sample_ratio 0.0010


python verdictdb/run.py --data_name pm25 --sample_ratio 0.0030
python verdictdb/run.py --data_name pm25 --sample_ratio 0.0028
python verdictdb/run.py --data_name pm25 --sample_ratio 0.0025
python verdictdb/run.py --data_name pm25 --sample_ratio 0.0023
python verdictdb/run.py --data_name pm25 --sample_ratio 0.0020
python verdictdb/run.py --data_name pm25 --sample_ratio 0.0018
python verdictdb/run.py --data_name pm25 --sample_ratio 0.0015