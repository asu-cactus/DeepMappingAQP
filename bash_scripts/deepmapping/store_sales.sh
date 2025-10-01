#!/bin/bash

# 5735.73 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 20e-4 --lr 1e-3 --milestone 500 1000

# 6190.34 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 17e-4 --lr 1e-3 --milestone 500 1000

# 6511.21 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 15e-4 --lr 1e-3 --milestone 500 1000

# 6881.81 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 13e-4 --lr 1e-3 --milestone 500 1000

# 7291.68 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 10e-4 --lr 1e-3 --milestone 500 1000

# 7503.86 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 8e-4 --lr 1e-3 --milestone 500 1000

# 7764.00 KB
python main.py --data_name store_sales --gpu 1 --allowed_error 5e-4 --lr 1e-3 --milestone 500 1000

# Insertion
python main.py --data_name store_sales --gpu 1 --allowed_error 20e-4 --lr 1e-3 --milestone 500 1000 --run_inserts
python main.py --data_name store_sales --gpu 1 --allowed_error 20e-4 --lr 1e-3 --milestone 500 1000 --run_inserts --uniform_update