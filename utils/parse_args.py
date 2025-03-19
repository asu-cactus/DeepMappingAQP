import argparse

resolution_dict = {
    "DEWP": 1,
    "TEMP": 1,
    "PRES": 1,
    "AT": 0.1,
    "AP": 0.1,
    "RH": 0.1,
    "DISTANCE": 0.1,
    "ARR_DELAY": 0.1,
    "list_price": 0.1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepMapping++ for AQP")
    # Data arguments
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument(
        "--ndim_input",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of input dimensions",
    )
    parser.add_argument(
        "--align", action="store_true", help="Align histogram with origin data"
    )

    # Auxilirary structure arguments
    parser.add_argument("--allowed_error", type=float, default=1e-4, help="Point error")
    parser.add_argument(
        "--output_scale",
        type=float,
        default=1000,
        help="range is [-output_scale, output_scale]",
    )
    parser.add_argument(
        "--sample_ratio", type=float, default=0.1, help="Sample ratio for training data"
    )
    # Training hyperparameters
    parser.add_argument("--units", type=int, default=200, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--print_every", type=int, default=100, help="Print every")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm")
    parser.add_argument("--milestones", type=int, nargs="+", default=[500, 1000])
    # Test arguments
    parser.add_argument("--synposis_only", action="store_true", help="Synopsis only")
    parser.add_argument("--nqueries", type=int, default=5000, help="Number of queries")
    parser.add_argument("--task_type", type=str, default="sum", help="Task type")
    # Update arguments
    parser.add_argument("--run_inserts", action="store_true", help="Run inserts")
    parser.add_argument(
        "--n_insert", type=int, default=10_000_000, help="Number of inserts"
    )
    parser.add_argument(
        "--n_insert_batch", type=int, default=4, help="Number of inserts"
    )
    parser.add_argument(
        "--no_retrain", action="store_true", help="No retrain after insert"
    )
    parser.add_argument(
        "--retrain_every_n_insert",
        type=int,
        default=2,
        help="Retrain every n insert batchs",
    )
    parser.add_argument(
        "--buffer_capacity", type=int, default=1000, help="Buffer capacity"
    )

    args = parser.parse_args()
    if args.ndim_input == 1:
        if args.data_name == "store_sales":
            args.indeps = ["list_price"]
            args.dep = "wholesale_cost"
        elif args.data_name == "flights":
            args.indeps = ["DISTANCE"]
            args.dep = "TAXI_OUT"
        elif args.data_name == "pm25":
            args.indeps = ["PRES"]
            args.dep = "pm25"
        elif args.data_name == "ccpp":
            args.indeps = ["RH"]
            args.dep = "PE"
        else:
            raise ValueError(f"No support for {args.data_name} for 1D input")
    else:
        if args.data_name == "pm25":
            args.indeps = ["TEMP", "PRES"]
            args.dep = "pm25"
        elif args.data_name == "ccpp":
            args.indeps = ["AT", "RH"]
            args.dep = "PE"
        elif args.data_name == "flights":
            args.indeps = ["ARR_DELAY", "DISTANCE"]
            args.dep = "TAXI_OUT"
        else:
            raise ValueError(f"No support for {args.data_name} for multi-dim input")

    args.resolutions = [resolution_dict[key] for key in args.indeps]

    print(args)
    return args
