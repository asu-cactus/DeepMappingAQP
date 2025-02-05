from parse_args import parse_args
from ml import get_model, train, test, create_aux_structure
from data_utils import read_data, prepare_training_data, get_dataloader
from time import time
from memory_profiler import memory_usage


def main():
    args = parse_args()
    df = read_data(args.data_name)
    X, y, X_scaler, y_scaler = prepare_training_data(
        df, args.indep, args.dep, args.resolution, args.output_scale
    )
    dataloader = get_dataloader(X, y, args.batch_size)
    model = get_model(args.units)
    train(model, dataloader, args.lr, args.epochs, args.print_every, args.gpu)
    aux_structure = create_aux_structure(
        model, X, y, y_scaler, args.allowed_error, args.output_scale, args.gpu
    )
    query_path = f"query/{args.data_name}_{args.indep}_sum.npz"

    X_min = df[args.indep].min()

    start_time = time()

    # test(
    #     args.nqueries,
    #     model,
    #     aux_structure,
    #     X_scaler,
    #     y_scaler,
    #     args.gpu,
    #     query_path,
    #     X_min,
    #     args.resolution,
    # )
    mem = max(
        memory_usage(
            (
                test,
                (
                    args.nqueries,
                    model,
                    aux_structure,
                    X_scaler,
                    y_scaler,
                    args.gpu,
                    query_path,
                    X_min,
                    args.resolution,
                ),
            )
        )
    )
    print(f"Time: {time() - start_time}")
    print("Maximum memory used for : {} MiB".format(mem))


if __name__ == "__main__":
    main()
