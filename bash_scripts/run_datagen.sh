#!/bin/bash

# python datagen/datagen.py -s 100000000 -x data/pm25/sample.csv -y data/pm25/sample.json -o data/pm25/dataset_sum.csv --scale 10000
# python datagen/datagen.py -s 100000000 -x data/ccpp/sample.csv -y data/ccpp/sample.json -o data/ccpp/dataset_sum.csv --scale 1000
# python datagen/datagen.py -s 100000000 -x data/flights/sample.csv -y data/flights/sample.json -o data/flights/dataset_sum.csv --scale 10
# python datagen/datagen.py -s 100000000 -x data/store_sales/sample.csv -y data/store_sales/sample.json -o data/store_sales/dataset_sum.csv --scale 1000

# python datagen/datagen.py -r 42 -s 100000000 -x data/pm25/sample.csv -y data/pm25/sample.json -o data/update_data/pm25/insert.csv --scale 10000
# python filter_update_data.py --data_name pm25
# python datagen/datagen.py -r 42 -s 100000000 -x data/ccpp/sample.csv -y data/ccpp/sample.json -o data/update_data/ccpp/insert.csv --scale 1000
python filter_update_data.py --data_name ccpp
# python datagen/datagen.py -r 42 -s 100000000 -x data/flights/sample.csv -y data/flights/sample.json -o data/update_data/flights/insert.csv --scale 10
python filter_update_data.py --data_name flights
# python datagen/datagen.py -r 42 -s 100000000 -x data/tpc-ds/sample.csv -y data/tpc-ds/sample.json -o data/update_data/tpc-ds/insert.csv --scale 1000
python filter_update_data.py --data_name store_sales