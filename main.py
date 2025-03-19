from utils.parse_args import parse_args
from run_query_gen import get_X_dimensions
from utils.data_utils import (
    prepare_training_data,
    get_dataloader,
    standardize_data,
)
from time import perf_counter


def main(args):
    if args.ndim_input == 1:
        from ml.ml_1dim import (
            get_model,
            train,
            test,
            create_aux_structure,
            test_with_inserts,
        )
    else:
        from ml.ml_2dim import get_model, train, test, create_aux_structure
    # Prepare training data
    prepare_start = perf_counter()

    X_train, y_train = prepare_training_data(args)
    # Compute X_min and total_sum before scaling/standardization
    X_min = X_train[0][0] if args.ndim_input == 1 else X_train[0]
    # X_max = X_train[-1][0] if args.ndim_input == 1 else X_train[-1]
    # total_sum = y_train[-1][0]

    if args.ndim_input > 1:
        X_range_dim1, X_range_dim2, dim1_n_resol, dim2_n_resol = get_X_dimensions(
            X_train, args.resolutions
        )
    else:
        dim2_n_resol = None

    X_train, y_train, X_scaler, y_scaler = standardize_data(
        X_train, y_train, args.output_scale
    )
    dataloader = get_dataloader(X_train, y_train, args.batch_size)

    # Train model
    model = get_model(args.units)
    model = train(args, model, dataloader)

    # Create aux structure
    output_size = 2 * args.output_scale

    aux_struct = create_aux_structure(
        args, model, X_train, y_train, y_scaler, output_size, dim2_n_resol=dim2_n_resol
    )

    print(f"Preparing time: {perf_counter() - prepare_start}")

    # Run test
    if args.run_inserts:
        test_with_inserts(
            args,
            model,
            aux_struct,
            X_scaler,
            y_scaler,
            X_min,
            dim2_n_resol=dim2_n_resol,
        )
    else:
        test(
            args,
            model,
            aux_struct,
            X_scaler,
            y_scaler,
            X_min,
            dim2_n_resol=dim2_n_resol,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
