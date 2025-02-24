import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pdb
from time import perf_counter
import os
import warnings

warnings.filterwarnings("error")


def get_model(hidden_size: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(1, hidden_size, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_size, 1, bias=True),
    )
    param_size = sum([p.nelement() * p.element_size() for p in model.parameters()])
    print(f"Model size: {param_size / 1024:.2f} KB")
    return model


def train(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float,
    epochs: int,
    print_every: int,
    gpu: int,
    saved_path,
    disable_tqdm,
) -> nn.Module:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # Load and return model if exist
    if os.path.exists(saved_path):
        model.load_state_dict(torch.load(saved_path, weights_only=True))
        model = model.to(device)
        return model

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

    # print(f"Last learning rate: {scheduler.get_last_lr()}")
    # Load the best model state dict
    model.load_state_dict(best_state_dict)

    # Save best model
    torch.save(best_state_dict, saved_path)
    return model


def create_aux_structure(
    args,
    model: nn.Module,
    X: np.array,
    y: np.array,
    y_scaler: MinMaxScaler,
    output_size: float,
    **kwargs,
):

    model.eval()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    aux_array = np.zeros_like(y)

    # with torch.no_grad():
    #     X = torch.tensor(X, dtype=torch.float32).to(device)
    #     y_pred = model(X).cpu().numpy()
    # # Compute point relative error
    # error = np.absolute(y_pred - y) / output_size

    # selected_index = error > args.allowed_error
    # print(f"Aux store rate: {selected_index.sum() / len(y):.4f}")
    # origin_y = y_scaler.inverse_transform(y)
    # aux_array[selected_index] = origin_y[selected_index]
    # aux_array = aux_array.reshape(-1)

    # Rewrite the above code to break X and y into batches to avoid memory error
    batch_size = 1024
    n_batches = len(y) // batch_size
    total_selected = 0
    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        X_batch = torch.tensor(X[start:end], dtype=torch.float32).to(device)
        y_batch = y[start:end]
        with torch.no_grad():
            y_pred = model(X_batch).cpu().numpy()
        error = np.absolute(y_pred - y_batch) / output_size
        selected_index = error > args.allowed_error
        origin_y = y_scaler.inverse_transform(y_batch)
        aux_array[start:end][selected_index] = origin_y[selected_index]
        total_selected += selected_index.sum()
    print(f"Aux store rate: {total_selected / len(y):.4f}")

    aux_array = aux_array.reshape(-1)
    return aux_array


def test(
    args,
    model: nn.Module,
    aux_structure: np.array,
    X_scaler: StandardScaler,
    y_scaler: MinMaxScaler,
    query_path: str,
    X_min: float,
    total_sum: float,
    **kwargs,
):

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load queries
    npzfile = np.load(query_path)
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][: args.nqueries]

        # The first column is the start of the query range and the second column is the end
        start = perf_counter()
        total_rel_error = 0.0
        total_error_II = 0.0
        for query in queries:
            X, y = query[:2], query[2]
            aux_indices = ((X - X_min) / args.resolutions[0]).round().astype(int)
            aux_out = aux_structure[aux_indices]
            X = X_scaler.transform(X.reshape(-1, 1))
            with torch.no_grad():
                X = torch.tensor(X, dtype=torch.float32).to(device)
                y_pred = model(X).cpu().numpy()
            y_pred = y_scaler.inverse_transform(y_pred).reshape(-1)

            y_hat = np.where(aux_out != 0, aux_out, y_pred)
            y_hat = y_hat[1] - y_hat[0]

            y_hat /= args.sample_ratio

            rel_error = np.absolute(y_hat - y) / (y + 1e-6)
            error_II = np.absolute(y_hat - y) / total_sum

            total_rel_error += rel_error
            total_error_II += error_II

        avg_rel_error = total_rel_error / len(queries)
        avg_error_II = total_error_II / len(queries)
        avg_time = (perf_counter() - start) / len(queries)
        print(f"{query_percent} query:  executed in {avg_time} seconds on average.")
        print(f"Avg rel error: {avg_rel_error:.4f}, Avg error II: {avg_error_II:.4f}")
