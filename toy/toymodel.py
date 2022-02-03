import torch
from monotonenorm import direct_norm, GroupSort, project_norm


class TimesN(torch.nn.Module):
    def __init__(self, n: float):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n * x


class ABS(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


def no_norm(x, **kwargs):
    return x


def get_model(unconstrained=False, dev=None, L=1):
    norm = no_norm if unconstrained else direct_norm
    activation = torch.nn.ReLU() if unconstrained else GroupSort(1)
    layer_conf = (1, 32, 32, 32, 1)
    nlayers = len(layer_conf) - 1

    layers = []
    for i in range(nlayers):
        kind_i = "two-inf" if i == 0 else "inf"
        lin = torch.nn.Linear(layer_conf[i], layer_conf[i + 1], bias=True)#i != nlayers - 1)
        layers.append(
            norm(lin, kind=kind_i, always_norm=False, alpha=L ** (1 / nlayers))
        )
        if i < len(layer_conf) - 2:
            layers.append(activation)
    return torch.nn.Sequential(*layers).to(dev or "cpu")
