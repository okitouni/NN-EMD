import torch
from monotonenorm import direct_norm, GroupSort


class TimesN(torch.nn.Module):
    def __init__(self, n: float):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n * x


def get_model(dev=None):
    size = 128
    return torch.nn.Sequential(
        direct_norm(torch.nn.Linear(2, size), kind="two-inf", always_norm=True),
        GroupSort(size),
        direct_norm(torch.nn.Linear(size, size), kind="inf", always_norm=True),
        GroupSort(size),
        direct_norm(torch.nn.Linear(size, size), kind="inf", always_norm=True),
        GroupSort(size),
        direct_norm(torch.nn.Linear(size, 1), kind="inf", always_norm=True),
        TimesN(1.0)
    ).to(dev or "cpu")
