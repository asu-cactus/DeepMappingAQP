from utils.aux_struct import AuxStruct

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pdb
from time import perf_counter
from typing import Union
import os
import warnings

warnings.filterwarnings("error")

AUX_EMPTY = -1
EPS = 1e-6


def get_model(hidden_size: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(2, hidden_size, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_size, 1, bias=True),
    )
    return model


def train(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float,
    epochs: int,
    print_every: int,
    gpu: int,
    saved_path: str,
    disable_tqdm: bool,
) -> nn.Module:
    # Load and return model if exist
    if os.path.exists(saved_path):
        model.load_state_dict(torch.load(saved_path, weights_only=True))
        return model

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    best_state_dict = None
    for epoch in tqdm(range(1, epochs + 1), disable=disable_tqdm):
        total_loss = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(X.to(device))
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Save the best model every epoch
        if total_loss < best_loss:
            best_loss = total_loss
            best_state_dict = model.state_dict()

        if epoch % print_every == 0:
            print(f"Epoch: {epoch}, loss: {total_loss / len(dataloader):.4f}")

    # Load the best model state dict
    model.load_state_dict(best_state_dict)

    # Save best model
    torch.save(model.state_dict(), saved_path)
    return model


def create_aux_structure(
    model: nn.Module,
    X: np.array,
    y: np.array,
    y_scaler: MinMaxScaler,
    allowed_error: float,
    output_size: float,
    gpu: int,
) -> AuxStruct:
    model.eval()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    aux_array = np.ones_like(y, dtype=np.float32) * AUX_EMPTY

    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X).cpu().numpy()
    # Compute point relative error
    error = np.absolute(y_pred - y) / output_size

    selected_index = error > allowed_error
    origin_y = y_scaler.inverse_transform(y)
    aux_array[selected_index] = origin_y[selected_index]
    aux_array = aux_array.reshape(-1)

    # Use to test if test() is correct
    origin_y = y_scaler.inverse_transform(y)
    aux_array = origin_y.reshape(-1)

    # return aux_array

    init_bit_list = [1 if aux != AUX_EMPTY else 0 for aux in aux_array]
    init_list = aux_array[aux_array != AUX_EMPTY]
    aux_struct = AuxStruct(init_bit_list, init_list)
    return aux_struct


def test(
    nqueries: int,
    model: nn.Module,
    aux_struct: Union[np.array, AuxStruct],
    X_scaler: StandardScaler,
    y_scaler: MinMaxScaler,
    gpu: int,
    query_path: str,
    X_min: list[float],
    total_sum: float,
    resolutions: list[float],
    dim2_n_resol: int,
):

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Load queries
    npzfile = np.load(query_path)
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][:nqueries]

        # The first column is the start of the query range and the second column is the end
        start = perf_counter()
        total_rel_error = 0.0
        total_error_II = 0.0
        for query in queries:
            X_start_d1, X_start_d2, X_end_d1, X_end_d2, y = query

            # Get output of the aux structure
            X_s_d1 = (X_start_d1 - X_min[0]) // resolutions[0]
            X_s_d2 = (X_start_d2 - X_min[1]) // resolutions[1]
            X_e_d1 = (X_end_d1 - X_min[0]) // resolutions[0]
            X_e_d2 = (X_end_d2 - X_min[1]) // resolutions[1]

            upper_right_index = int(X_e_d1 * dim2_n_resol + X_e_d2)
            lower_right_index = int(X_e_d1 * dim2_n_resol + X_s_d2)
            upper_left_index = int(X_s_d1 * dim2_n_resol + X_e_d2)
            lower_left_index = int(X_s_d1 * dim2_n_resol + X_s_d2)

            # aux_outs = [
            #     aux_struct[upper_right_index],
            #     aux_struct[lower_right_index],
            #     aux_struct[upper_left_index],
            #     aux_struct[lower_left_index],
            # ]
            aux_outs = [
                aux_struct.get(index, AUX_EMPTY)
                for index in [
                    upper_right_index,
                    lower_right_index,
                    upper_left_index,
                    lower_left_index,
                ]
            ]

            # Get the prediction from the model
            X = np.array(
                [
                    [X_end_d1, X_end_d2],
                    [X_end_d1, X_start_d2],
                    [X_start_d1, X_end_d2],
                    [X_start_d1, X_start_d2],
                ]
            )
            X = X_scaler.transform(X)
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32).to(device)
                y_pred = model(X).cpu().numpy()
            y_preds = y_scaler.inverse_transform(y_pred).reshape(-1)

            # Combine the prediction and the aux structure
            y_hat = [
                y_pred if aux_out == AUX_EMPTY else aux_out
                for aux_out, y_pred in zip(aux_outs, y_preds)
            ]

            # y_hat = [
            #     y_pred if aux_out is None else aux_out
            #     for aux_out, y_pred in zip(aux_outs, y_preds)
            # ]

            # Compuate final prediction
            y_hat = y_hat[0] - y_hat[1] - y_hat[2] + y_hat[3]

            # y_hat = aux_outs[0] - aux_outs[1] - aux_outs[2] + aux_outs[3]

            rel_error = np.absolute(y_hat - y) / (y + EPS)
            error_II = np.absolute(y_hat - y) / total_sum

            total_rel_error += rel_error
            total_error_II += error_II

        avg_rel_error = total_rel_error / len(queries)
        avg_error_II = total_error_II / len(queries)
        avg_time = (perf_counter() - start) / len(queries)
        print(f"{query_percent} query:  executed in {avg_time} seconds on average.")
        print(f"Avg rel error: {avg_rel_error:.6f}, Avg error II: {avg_error_II:.9f}\n")
