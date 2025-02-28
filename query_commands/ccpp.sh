#!/bin/bash


# 849.99 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 8e-4 --epochs 2000

# 1261.38 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 5e-4 --epochs 2000

# 1771.81 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 3e-4 --epochs 2000

# 2228.87 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 2e-4 --epochs 2000

# 2679.69 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 1e-4 --epochs 2000

# 2778.50 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 0.8e-4 --epochs 2000

# 2935.93 KB
python main.py --data_name ccpp --units 200 --gpu 0 --lr 2e-4 --milestones 300 700 --allowed_error 0.5e-4 --epochs 2000