from parse_args import parse_args
from query_gen import get_X_dimensions

from utils.data_utils import (
    prepare_training_data,
    get_dataloader,
    standardize_data,
)
from time import perf_counter


def main():
    args = parse_args()
    if args.ndim_input == 1:
        from ml_1dim import get_model, train, test, create_aux_structure
    else:
        from ml_2dim import get_model, train, test, create_aux_structure

    # Prepare training data

    prepare_start = perf_counter()
    X, y = prepare_training_data(
        args.data_name,
        args.task_type,
        args.indeps,
        args.dep,
        args.resolutions,
        args.ndim_input,
    )
    # Compute X_min and total_sum before scaling
    X_min = X[0][0] if args.ndim_input == 1 else X[0]
    total_sum = y[-1][0]
    X_range_dim1, X_range_dim2, dim1_n_resol, dim2_n_resol = get_X_dimensions(
        X, args.resolutions
    )

    X, y, X_scaler, y_scaler = standardize_data(X, y, args.output_scale)
    dataloader = get_dataloader(X, y, args.batch_size)

    # Train model
    model = get_model(args.units)
    saved_path = (
        f"saved_models/{args.data_name}_{args.task_type}_{args.ndim_input}D.pth"
    )
    train(
        model, dataloader, args.lr, args.epochs, args.print_every, args.gpu, saved_path
    )

    # Create aux structure
    output_size = 2 * args.output_scale
    aux_structure = create_aux_structure(
        model, X, y, y_scaler, args.allowed_error, output_size, args.gpu
    )
    print(f"Preparing time: {perf_counter() - prepare_start}")

    # Run test
    query_path = f"query/{args.data_name}_{args.task_type}_{args.ndim_input}D.npz"

    test(
        args.nqueries,
        model,
        aux_structure,
        X_scaler,
        y_scaler,
        args.gpu,
        query_path,
        X_min,
        total_sum,
        args.resolutions,
        dim2_n_resol,
    )


if __name__ == "__main__":
    main()
