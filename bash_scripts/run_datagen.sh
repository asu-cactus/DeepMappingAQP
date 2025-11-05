#!/bin/bash
python datagen/prepare_pm25_for_datagen.py
python datagen/prepare_ccpp_for_datagen.py
python datagen/prepare_flights_for_datagen.py
python datagen/prepare_tpc_for_datagen.py

python datagen/datagen.py -s 100000000 -x data/pm25/sample.csv -y data/pm25/sample.json -o data/pm25/dataset_sum.csv --scale 10000
python datagen/datagen.py -s 100000000 -x data/ccpp/sample.csv -y data/ccpp/sample.json -o data/ccpp/dataset_sum.csv --scale 1000
python datagen/datagen.py -s 100000000 -x data/flights/sample.csv -y data/flights/sample.json -o data/flights/dataset_sum.csv --scale 10
python datagen/datagen.py -s 100000000 -x data/tpc-ds/sample.csv -y data/tpc-ds/sample.json -o data/tpc-ds/dataset_sum.csv --scale 1000
python datagen/datagen.py -s 100000000 -x data/part/sample.csv -y data/part/sample.json -o data/part/dataset_sum.csv --scale 10
python datagen/datagen.py -s 100000000 -x data/lineitem/sample.csv -y data/lineitem/sample.json -o data/lineitem/dataset_sum.csv --scale 1

python datagen/datagen.py -r 42 -s 100000000 -x data/pm25/sample.csv -y data/pm25/sample.json -o data/update_data/pm25/insert_origin.csv --scale 10000
python filter_update_data.py --data_name pm25
python filter_update_data.py --data_name pm25 --uniform_update
python datagen/datagen.py -r 42 -s 100000000 -x data/ccpp/sample.csv -y data/ccpp/sample.json -o data/update_data/ccpp/insert_origin.csv --scale 1000
python filter_update_data.py --data_name ccpp
python filter_update_data.py --data_name ccpp --uniform_update
python datagen/datagen.py -r 42 -s 100000000 -x data/flights/sample.csv -y data/flights/sample.json -o data/update_data/flights/insert_origin.csv --scale 10
python filter_update_data.py --data_name flights
python filter_update_data.py --data_name flights --uniform_update
python datagen/datagen.py -r 42 -s 100000000 -x data/tpc-ds/sample.csv -y data/tpc-ds/sample.json -o data/update_data/tpc-ds/insert_origin.csv --scale 1000
python filter_update_data.py --data_name store_sales
python filter_update_data.py --data_name store_sales --uniform_update