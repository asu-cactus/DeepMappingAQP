#!/bin/bash

# 1107.44 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 7e-4 

# 1243.12 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 5e-4 

# 1603.84 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 3e-4  

# 1834.80 KB 
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 2e-4      

# 2004.13 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 1e-4 

# 2039.19 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 8e-5

# 2085.51 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 5e-5

# 2116.53 KB
python main.py --data_name flights --units 200 --gpu 0 --lr 5e-4 --milestones 200 800 --allowed_error 3e-5  