#!/bin/bash

# 5847.28 KB
python main.py --data_name part --gpu 1 --allowed_error 20e-4 --lr 1e-3 --milestone 500 1000

# 6085.01 KB
python main.py --data_name part --gpu 0 --allowed_error 10e-4 --lr 1e-3 --milestone 500 1000

# 6307.84 KB
python main.py --data_name part --gpu 0 --allowed_error 9e-4 --lr 1e-3 --milestone 500 1000

# 6578.75 KB
python main.py --data_name part --gpu 0 --allowed_error 8e-4 --lr 1e-3 --milestone 500 1000

# 6807.57 KB
python main.py --data_name part --gpu 0 --allowed_error 7e-4 --lr 1e-3 --milestone 500 1000

# 7010.03 KB
python main.py --data_name part --gpu 0 --allowed_error 6e-4 --lr 1e-3 --milestone 500 1000

# 7438.41 KB
python main.py --data_name part --gpu 0 --allowed_error 4e-4 --lr 1e-3 --milestone 500 1000

# 7857.89 KB
python main.py --data_name part --gpu 0 --allowed_error 2e-4 --lr 1e-3 --milestone 500 1000