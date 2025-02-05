import argparse

resolution_dict = {
    "DEWP": 1,
    "TEMP": 1,
    "PRES": 1,
    "AT": 0.01,
    "AP": 0.01,
    "RH": 0.01,
    "DISTANCE": 1,
    "list_price": 0.05,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepMapping++ for AQP")
    # Data arguments
    parser.add_argument("--data_name", type=str, required=True, help="Data name")
    parser.add_argument("--indep", type=str, help="Independent variable")
    parser.add_argument("--dep", type=str, help="Dependent variable")
    # Auxilirary structure arguments
    parser.add_argument("--allowed_error", type=float, default=5e-4, help="Point error")
    parser.add_argument(
        "--output_scale",
        type=float,
        default=1000,
        help="range is [-output_scale, output_scale]",
    )
    # Training hyperparameters
    parser.add_argument("--units", type=int, default=500, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument("--print_every", type=int, default=500, help="Print every")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    # Test arguments
    parser.add_argument("--nqueries", type=int, default=1000, help="Number of queries")

    args = parser.parse_args()
    if args.data_name == "store_sales":
        args.indep = "list_price"
        args.dep = "wholesale_cost"
    elif args.data_name == "flights":
        args.indep = "DISTANCE"
        args.dep = "TAXI_OUT"
    elif args.data_name == "pm25":
        args.dep = "pm2.5"
    elif args.data_name == "ccpp":
        args.dep = "PE"

    if not args.indep:
        raise ValueError("Please specify --indep")
    args.resolution = resolution_dict[args.indep]

    print(args)
    return args
