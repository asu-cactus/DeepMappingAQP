from dbestclient.executor.executor import SqlExecutor
import numpy as np
import pandas as pd

import pdb
import argparse
import subprocess
import os
from time import perf_counter

EPS = 1e-6
n_insert_batch = 4
sample_ratio = 0.1

def parse_args():
    parser = argparse.ArgumentParser(description="Create a sample of a dataset")
    parser.add_argument("--data_name", type=str, required=True, help="The name of the dataset")
    parser.add_argument("--units", type=int, default=200, help="The number of units to sample")
    parser.add_argument("--use_existing_model", action="store_true", help="Use existing model")
    parser.add_argument("--run_inserts", action="store_true", help="Run insert workload")
    parser.add_argument("--nqueries", type=int, default=5000, help="Number of queries to run")
    parser.add_argument(
        "--retrain_every_n_insert",
        type=int,
        default=2,
        help="Retrain every n insert batchs",
    )
    args = parser.parse_args()
    return args 

def combine_origin_and_insert_data(origin_sample, insert_df, ith_insert, combined_data_path):
    batch_size = len(insert_df) // n_insert_batch
    if ith_insert > 0:
        insert_df = insert_df[: ith_insert * batch_size]
        insert_df = insert_df.sample(frac=sample_ratio, random_state=42)
        combined_df = pd.concat([origin_sample, insert_df])
    else:
        combined_df = origin_sample
    combined_df.to_csv(combined_data_path, index=False)

def execute_shell_command(command):
    try:
        process = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        return None, e.stderr
    
def move_model_files(to_path):    
    command_to_execute = f"mkdir -p ../dbestwarehouse_temp/{to_path}/"
    execute_shell_command(command_to_execute)
    command_to_execute = f"mv ../dbestwarehouse/*.dill ../dbestwarehouse_temp/{to_path}/"
    execute_shell_command(command_to_execute)
    command_to_execute = f"mv ../dbestwarehouse/*.pt ../dbestwarehouse_temp/{to_path}/"
    execute_shell_command(command_to_execute)

class Query1:
    def __init__(self, args):
        self.sql_executor = SqlExecutor()

        self.task_type = "sum"
 
        self.units = args.units
        self.data_name = args.data_name
        self.nqueries = args.nqueries   

        if args.data_name == "pm25":
            self.dep = "pm25"
            self.indep = "PRES"
        elif args.data_name == "ccpp":
            self.dep = "PE"
            self.indep = "RH"
        elif args.data_name == "flights":
            self.dep = "TAXI_OUT"
            self.indep = "DISTANCE"
        elif args.data_name == "store_sales":
            self.dep = "wholesale_cost"
            self.indep = "list_price"
        else:
            raise ValueError(f"Invalid data name: {args.data_name}")

        self.mdl_name = f"{self.data_name}_{self.units}"
        self.datafile = f"{args.data_name}_sample.csv"
        # self.query_path = f"../../DeepMappingAQP/query/{self.data_name}_{self.task_type}_1D.npz"
        self.query_path = f"../../DeepMappingAQP/query/{self.data_name}_{self.task_type}_1D_nonzeros.npz"
        self.insert_query_path = f"../../DeepMappingAQP/query/{self.data_name}_insert_1D_nonzeros.npz"

    def build_model(self):
        self.sql_executor.execute(f"set n_mdn_layer_node_reg={self.units}")          # 20
        self.sql_executor.execute(f"set n_mdn_layer_node_density={self.units}")      # 30
        self.sql_executor.execute("set n_hidden_layer=2")                 # 2
        self.sql_executor.execute("set n_gaussians_reg=10")                # 
        self.sql_executor.execute("set n_gaussians_density=10")            # 10
        self.sql_executor.execute("set n_epoch=10")                       # 20
        self.sql_executor.execute("set device='gpu'")

        
        start = perf_counter()
        self.sql_executor.execute(
            f"create table {self.mdl_name}({self.dep} real, {self.indep} real) from {self.datafile} method uniform size {sample_ratio}"
        )
        train_time = (perf_counter() - start) / 60
        print(f"Model training time: {train_time} minutes")
        with open(f"results/{self.data_name}_train_time.csv", "a") as f:
            f.write(f"{self.units},{train_time}\n")
     
    def workload(
        self,
        n_jobs: int = 1,
    ):
        
        self.sql_executor.execute("set n_jobs=" + str(n_jobs) + '"')
        # Load dataframe from save_path if exists
        save_path = f"results/{self.data_name}_results.csv"
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=["units", "query_percent", "avg_rel_err", "avgtime"])

        # Load queries
        results = []
        
        npzfile = np.load(self.query_path)
        for query_percent in npzfile.keys():
            queries = npzfile[query_percent][:self.nqueries]
            start = perf_counter()
            total_rel_error = 0.0
  
            for query in queries:
                lb, ub, y = query

                y_pred = self.sql_executor.execute(
                    f"select {self.task_type}({self.dep} real) from {self.mdl_name} where {lb}<{self.indep}<={ub}"
                )["dummy"]
                rel_error = np.absolute(y_pred - y) / (y + EPS)
                total_rel_error += rel_error

            avg_rel_error = total_rel_error / len(queries)
            avg_time = (perf_counter() - start) / len(queries)
            results.append({"units": self.units, "query_percent": query_percent,"avg_rel_err": round(avg_rel_error,4), "avgtime": round(avg_time,4)})

            move_model_files(self.data_name)
        # Append results to dataframe
        df = df.append(results, ignore_index=True)
        df.to_csv(save_path, index=False)


    def insert_workload(
            self,
            n_jobs: int = 1,
            nqueries: int = 1000,
        ):
        self.sql_executor.execute("set n_jobs=" + str(n_jobs) + '"')
        origin_sample = pd.read_csv(f"../dbestwarehouse_temp/{self.data_name}_{self.task_type}_sample.csv")
        insert_df = pd.read_csv(f"../dbestwarehouse_temp/{self.data_name}_insert.csv")

        combined_data_path = f"../dbestwarehouse/{self.datafile}"

        # Load dataframe from save_path if exists
        save_path = f"results/{self.data_name}_insert_results.csv"
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
        else:
            df = pd.DataFrame(columns=["query_percent", "ith_insert", "avg_rel_err", "avgtime"])
        results = []
    
        npzfile = np.load(self.insert_query_path)
        for query_percent, query_group in npzfile.items():
            for i, queries in enumerate(query_group):
                # Combine origin and insert data
                combine_origin_and_insert_data(origin_sample, insert_df, i, combined_data_path)
                if i % args.retrain_every_n_insert == 0:
                    self.build_model()

                # Start workload
                queries = queries[: nqueries]
                start = perf_counter()
                total_rel_error = 0.0
                for query in queries:
                    lb, ub, y = query[0], query[1], query[2]
                    y_pred = self.sql_executor.execute(
                        f"select {self.task_type}({self.dep} real) from {self.mdl_name} where {lb}<{self.indep}<={ub}"
                    )["dummy"]
                    rel_error = np.absolute(y_pred - y) / (y + EPS)
                    total_rel_error += rel_error
                avg_rel_error = total_rel_error / len(queries)
                avg_time = (perf_counter() - start) / len(queries)
                results.append({"query_percent": query_percent, "ith_insert": i, "avg_rel_err": round(avg_rel_error,4), "avgtime": round(avg_time,4)})
                print(f"Avg. relative error: {avg_rel_error}, Avg. time: {avg_time}")   
                if (i + 1) % args.retrain_every_n_insert == 0:
                    nth_update = (i + 1) // args.retrain_every_n_insert - 1
                    move_model_files(f"insert_models/{self.data_name}/{nth_update}")
        # Append results to dataframe
        df = df.append(results, ignore_index=True)
        df.to_csv(save_path, index=False)

        # Remove the combined data file
        execute_shell_command(f"rm {combined_data_path}")

            
if __name__ == "__main__":
    args  = parse_args()
    if args.use_existing_model:
        execute_shell_command(f"cp ../dbestwarehouse_temp/{args.data_name}/{args.data_name}_{args.units}* ../dbestwarehouse/")
    query1 = Query1(args)

    if args.run_inserts:
        query1.insert_workload()
    else:
        execute_shell_command(f"cp ../dbestwarehouse_temp/{args.data_name}_{query1.task_type}_sample.csv ../dbestwarehouse/{query1.datafile}")
        if not args.use_existing_model:
            query1.build_model()
        query1.workload()    
        execute_shell_command(f"rm ../dbestwarehouse/{query1.datafile}")