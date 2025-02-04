import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import pdb


def get_model(hidden_size: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(1, hidden_size, bias=True),
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
) -> nn.Module:
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model.train()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    best_state_dict = None
    for epoch in tqdm(range(1, epochs + 1)):
        total_loss = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(X.to(device))
            loss = criterion(outputs, y.to(device))
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = model.state_dict()

            if epoch % print_every == 0:
                total_loss += loss.item()

        if epoch % print_every == 0:
            print(f"Epoch: {epoch}, loss: {total_loss / len(dataloader):.4f}")

    # print(f"Last learning rate: {scheduler.get_last_lr()}")
    # Load the best model state dict
    model.load_state_dict(best_state_dict)
    return model


def create_aux_structure(
    model: nn.Module,
    X: np.array,
    y: np.array,
    y_scaler: MinMaxScaler,
    allowed_error: float,
    output_scale: float,
    gpu: int,
):
    model.eval()
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    aux_array = np.zeros_like(y)
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X).cpu().numpy()
    # Compute point relative error
    error = np.absolute(y_pred - y) / (2 * output_scale)

    selected_index = error > allowed_error
    origin_y = y_scaler.inverse_transform(y)
    aux_array[selected_index] = origin_y[selected_index]
    aux_array = aux_array.reshape(-1)
    pdb.set_trace()

    return aux_array


def test(
    nqueries: int,
    model: nn.Module,
    aux_structure: np.array,
    X_scaler: StandardScaler,
    y_scaler: MinMaxScaler,
    gpu: int,
    query_path: str,
):

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Load queries
    npzfile = np.load(query_path)
    for query_percent in npzfile.keys():
        queries = npzfile[query_percent][:nqueries]

        # The first column is the start of the query range and the second column is the end
        X = np.concatenate((queries[:, 0], queries[:, 1]), axis=0)
        pdb.set_trace()
        X = X_scaler.transform(X)

        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(device)
            y_pred = model().cpu().numpy()
        y_pred = y_scaler.inverse_transform(y_pred).reshape(-1)
        y_pred = y_pred[nqueries:] - y_pred[:nqueries]

        y_hat = np.where(aux_structure != 0, aux_structure, y_pred)  # wrong
        # Compute the relative error
        y = queries[:, 2].reshape(-1)
        rel_errors = np.absolute(y_hat - y) / y
        avg_rel_error = np.mean(rel_errors)
        print(
            f"Query percent: {query_percent}, Average relative error: {avg_rel_error:.4f}"
        )

        pdb.set_trace()
