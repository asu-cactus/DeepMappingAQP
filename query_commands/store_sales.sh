#!/bin/bash

# 2210.94 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 5e-3 --lr 1e-3 --milestone 500 1000

# 2676.39 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 4e-3 --lr 1e-3 --milestone 500 1000 

# 3757.44 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 3e-3 --lr 1e-3 --milestone 500 1000

# 5736.66 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 20e-4 --lr 1e-3 --milestone 500 1000

# 6510.06 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 15e-4 --lr 1e-3 --milestone 500 1000

# 7292.04 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 10e-4 --lr 1e-3 --milestone 500 1000

# 7502.74 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 8e-4 --lr 1e-3 --milestone 500 1000

# 7762.80 KB
python main.py --data_name store_sales --units 200 --gpu 1 --allowed_error 5e-4 --lr 1e-3 --milestone 500 1000