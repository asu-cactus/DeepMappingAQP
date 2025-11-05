#!/bin/bash

# 2854.64 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 16e-4 --lr 1e-3 --milestone 500 1000

# 2996.76 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 14e-4 --lr 1e-3 --milestone 500 1000

# 3201.38 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 12e-4 --lr 1e-3 --milestone 500 1000

# 3434.50 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 10e-4 --lr 1e-3 --milestone 500 1000

# 3641.11 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 8e-4 --lr 1e-3 --milestone 500 1000

# 3809.31 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 6e-4 --lr 1e-3 --milestone 500 1000

# 3978.33 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 4e-4 --lr 1e-3 --milestone 500 1000

# 4148.63 KB
python main.py --data_name lineitem --gpu 1 --allowed_error 2e-4 --lr 1e-3 --milestone 500 1000
