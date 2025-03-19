#!/bin/bash

# (2148.44/4+1069.32)/800 = 2.008025
python verdictdb/run.py --do_insert --data_name pm25 --sample_ratio 0.003


# (2914.05/4+849.86)/800 = 1.972
python verdictdb/run.py --do_insert --data_name ccpp --sample_ratio 0.003


# (1941.97/4+1603.84)/800 = 2.6116
python verdictdb/run.py --do_insert --data_name flights --sample_ratio 0.004

# (7773.42/4+5735.73)/800 = 9.5988
python verdictdb/run.py --do_insert --data_name store_sales --sample_ratio 0.012

