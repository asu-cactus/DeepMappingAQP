#!/bin/bash

python run_deepmapping_variants.py --dm_variant NHP --dataset pm25
python run_deepmapping_variants.py --dm_variant NHP --dataset ccpp
python run_deepmapping_variants.py --dm_variant NHP --dataset flights
python run_deepmapping_variants.py --dm_variant NHP --dataset store_sales

python run_deepmapping_variants.py --dm_variant NHR --dataset pm25
python run_deepmapping_variants.py --dm_variant NHR --dataset ccpp
python run_deepmapping_variants.py --dm_variant NHR --dataset flights
python run_deepmapping_variants.py --dm_variant NHR --dataset store_sales