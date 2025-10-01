#!/bin/bash

# 2294.1 KB
python main.py --data_name pm25 --gpu 1 --allowed_error 0.2e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1998.71 KB
python main.py --data_name pm25 --gpu 1 --allowed_error 1e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1652.82 KB
python main.py --data_name pm25 --gpu 1 --allowed_error 2e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1411.70 KB
python main.py --data_name pm25 --gpu 1 --allowed_error 3e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1228.40 KB
python main.py --data_name pm25 --gpu 1 --allowed_error 4e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1069.32 KB
python main.py --data_name pm25 --gpu 1 --allowed_error 5e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000


# Insertion
python main.py --data_name pm25 --gpu 1 --allowed_error 5e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000 --run_inserts
python main.py --data_name pm25 --gpu 1 --allowed_error 5e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000 --run_inserts --uniform_update