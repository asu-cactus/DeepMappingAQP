import torch.nn as nn
import torch


def get_model(device, hidden_size):
    model = nn.Sequential(
        nn.Linear(1, hidden_size, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_size, 1, bias=True),
    ).to(device)
