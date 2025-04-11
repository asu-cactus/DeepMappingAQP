from utils.parse_args import parse_args
from utils.data_utils import (
    get_dataloader,
    standardize_data,
    prepare_training_data_for_NHR,
    prepare_training_data_for_NHP,
)
from time import perf_counter
from ml.ml_1dim import get_model, train, create_aux_structure, test_NHP, test_NHR


def run(args):

    # Prepare training data
    prepare_start = perf_counter()

    if args.dm_variant == "NHP":
        X_train, y_train = prepare_training_data_for_NHP(args)
    elif args.dm_variant == "NHR":
        X_train, y_train, train_data_lens = prepare_training_data_for_NHR(args)
    else:
        raise ValueError(f"Unknown dm_variant: {args.dm_variant}")

    # Compute X_min and total_sum before scaling/standardization
    X_min = X_train[0]
    X_max = X_train[-1]

    X_train, y_train, X_scaler, y_scaler = standardize_data(
        X_train, y_train, args.output_scale
    )
    dataloader = get_dataloader(X_train, y_train, args.batch_size)

    # Train model
    in_units = 1 if args.dm_variant != "NHR" else 2
    model = get_model(args.units, in_units=in_units)
    model = train(args, model, dataloader, variant=args.dm_variant)

    # Create aux structure
    output_size = 2 * args.output_scale

    aux_struct = create_aux_structure(
        args, model, X_train, y_train, y_scaler, output_size
    )

    print(f"Preparing time: {perf_counter() - prepare_start}")

    # Run test
    if args.dm_variant == "NHP":
        test_NHP(args, model, aux_struct, X_scaler, y_scaler, X_min)
    elif args.dm_variant == "NHR":
        X_min = X_min[0]
        X_max = X_max[0]
        X_range = X_max - X_min
        test_NHR(
            args, model, aux_struct, X_scaler, y_scaler, X_min, X_range, train_data_lens
        )
    else:
        raise ValueError(f"Unknown dm_variant: {args.dm_variant}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
