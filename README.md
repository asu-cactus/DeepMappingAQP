- [Software Installation](#software-installation)
  - [Set up PyMySQL](#set-up-pymysql)
- [Datasets](#datasets)
- [Run experiments](#run-experiments)
  - [Scale data set](#scale-data-set)
  - [Generate range queries](#generate-range-queries)
  - [Run DeepMapping-R](#run-deepmapping-r)
- [Run baselines](#run-baselines)
  - [VerdictDB](#verdictdb)
  - [DBEst++](#dbest)
  - [Synopsis](#synopsis)
  - [DeepMapping-R variants](#deepmapping-r-variants)
  
  
## Software Installation
Note that the software requirement for DeepMapping-R is different from VerdictDB and DBEst++. For [VerdictDB](https://docs.verdictdb.org/documentation/quickstart/quickstart/) and [DBEst++](https://github.com/qingzma/DBEst_MDN), please refer to their original repositories for installation instructions.
For VerdictDB, we used python API and pymyql. According to their document, python 3.7 is required in conda environment.

For DeepMapping-R, install 
```
python -m pip install -r requirements.txt
```

### Set up PyMySQL
To set up default password for root user, follow [here](https://stackoverflow.com/questions/39281594/error-1698-28000-access-denied-for-user-rootlocalhost#:~:text=system_user%20(recommended)-,Option%201%3A,-sudo%20mysql%20%2Du):
```
sudo mysql -u root # I had to use "sudo" since it was a new installation

mysql> USE mysql;
mysql> UPDATE user SET plugin='mysql_native_password' WHERE User='root';
mysql> FLUSH PRIVILEGES;
mysql> exit;

sudo service mysql restart
```


## Datasets
Datasets are publicly available from the web. You can download them via the following links.

[PM25](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data), [CCPP](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant), [Flights](https://www.kaggle.com/datasets/usdot/flight-delays), [TPC-DS](https://www.tpc.org/tpcds/)

We generated the Store Sales table with the TPC-DS tools with SF=1.0.
```
dsqgen --SCALE 1 --OUTPUT_DIR tpcds_data_SF1
```


## Run experiments
### Scale data set 
The dataset is first scale up using [IDEBench](https://github.com/IDEBench/IDEBench-public).

```
bash bash_scipts/run_datagen.sh
```

### Generate range queries
After data is prepared, generated the range queries:
```
bash bash_scripts/deepmapping/run_query_gen.sh
```

### Run DeepMapping-R
Run DeepMapping-R:
```
bash bash_scripts/deepmapping/pm25.sh
bash bash_scripts/deepmapping/ccpp.sh
bash bash_scripts/deepmapping/flights.sh
bash bash_scripts/deepmapping/store_sales.sh
```

## Run baselines
There are five baselines: VerdictDB, DBEst++, Synopsis, two DeepMapping Variants

### VerdictDB
First, insert data into MySQL database 
```
bash bash_scripts/verdictdb/add_data.sh
```
Then you can run the queries with different sample ratios
```
bash bash_scripts/verdictdb/diff_sample_ratio.sh
```
You can run the insertion and then query.
```
bash bash_scripts/verdictdb/insert.sh
```

### DBEst++
You can run DBEst++ by downloading their repository, and copy all files in "DBEst++" in our repository to "experiments" folder. You can run the queries:
```
bash run_train.sh
```
To run the insertion experiment:
```
bash run_insert.sh
```
If you want to know the model size, you can get a model size chart by running
```
python get_model_sizes.py
```

### Synopsis
"Synopsis" is directly using the training data of DeepMapping-R to answer range queries.
```
bash bash_scripts/deepmapping/run_synopsis.py
```

### DeepMapping-R variants
Run two DeepMapping-R variants NHP and NHR:
```
bash bash_scripts/deepmapping/run_deepmapping_variants.py
```

