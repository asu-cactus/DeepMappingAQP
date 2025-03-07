#!/bin/bash

# 2294.14 KB
python main.py --data_name pm25 --units 200 --gpu 1 --allowed_error 0.2e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 2219.31 KB
python main.py --data_name pm25 --units 200 --gpu 1 --allowed_error 0.5e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 2181.69 KB
python main.py --data_name pm25 --units 200 --gpu 1 --allowed_error 0.5e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1998.70 KB
python main.py --data_name pm25 --units 200 --gpu 1 --allowed_error 1e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1652.83 KB
python main.py --data_name pm25 --units 200 --gpu 1 --allowed_error 2e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000

# 1069.31 KB
python main.py --data_name pm25 --units 200 --gpu 1 --allowed_error 5e-3 --lr 1e-4 --epochs 2000 --milestone 1000 2000