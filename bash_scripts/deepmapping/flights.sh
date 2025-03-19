#!/bin/bash

# # 1107.46 KB
# python main.py --data_name flights --gpu 0 --allowed_error 7e-4  --lr 5e-4 --milestones 200 800

# # 1243.12 KB
# python main.py --data_name flights --gpu 0 --allowed_error 5e-4  --lr 5e-4 --milestones 200 800

# # 1362.21 KB
# python main.py --data_name flights --gpu 0 --allowed_error 4e-4  --lr 5e-4 --milestones 200 800

# # 1603.84 KB
# python main.py --data_name flights --gpu 0 --allowed_error 3e-4  --lr 5e-4 --milestones 200 800

# # 1834.69 KB
# python main.py --data_name flights --gpu 0 --allowed_error 2e-4  --lr 5e-4 --milestones 200 800

# # 2004.21 KB
# python main.py --data_name flights --gpu 0 --allowed_error 1e-4  --lr 5e-4 --milestones 200 800

# # 2116.53 KB
# python main.py --data_name flights --gpu 0 --allowed_error 0.3e-4 --lr 5e-4 --milestones 200 800

# Insertion
python main.py --data_name flights --gpu 0 --allowed_error 3e-4  --lr 5e-4 --milestones 200 800 --run_inserts