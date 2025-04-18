#!/bin/bash

# 849.86 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 8e-4 --lr 2e-4 --milestones 300 700 --epochs 2000

# 1260.94 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 5e-4 --lr 2e-4 --milestones 300 700  --epochs 2000

# 1771.61 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 3e-4 --lr 2e-4 --milestones 300 700  --epochs 2000

# 2228.83 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 2e-4 --lr 2e-4 --milestones 300 700  --epochs 2000

# 2680.00 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 1e-4 --lr 2e-4 --milestones 300 700  --epochs 2000

# 2935.49 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 0.5e-4 --lr 2e-4 --milestones 300 700  --epochs 2000

# 3116.18 KB
python main.py --data_name ccpp --gpu 0 --allowed_error 0.1e-4 --lr 2e-4 --milestones 300 700  --epochs 2000

# Insertion
python main.py --data_name ccpp --gpu 0 --allowed_error 8e-4 --lr 2e-4 --milestones 300 700 --epochs 2000 --run_inserts