import argparse

resolution_dict = {
    "DEWP": 1,
    "TEMP": 1,
    "PRES": 1,
    "AT": 0.02,
    "AP": 0.02,
    "RH": 0.02,
    "DISTANCE": 1,
    "ARR_DELAY": 1,
    "list_price": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepMapping++ for AQP")
    # Data arguments
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument(
        "--ndim_input",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of input dimensions",
    )
    # parser.add_argument(
    #     "--indeps", type=str, help="Independent variable, comma separated"
    # )
    # parser.add_argument("--dep", type=str, help="Dependent variable")

    # Auxilirary structure arguments
    parser.add_argument("--allowed_error", type=float, default=1e-4, help="Point error")
    parser.add_argument(
        "--output_scale",
        type=float,
        default=1000,
        help="range is [-output_scale, output_scale]",
    )
    # Training hyperparameters
    parser.add_argument(
        "--units", type=int, default=2000, help="Number of hidden units"
    )
    parser.add_argument("--epochs", type=int, default=20000, help="Number of epochs")
    parser.add_argument("--print_every", type=int, default=2000, help="Print every")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm")
    # Test arguments
    parser.add_argument("--nqueries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--task_type", type=str, default="sum", help="Task type")

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
