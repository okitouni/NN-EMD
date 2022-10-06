import torch
from monotonenorm import direct_norm, GroupSort

class TimesN(torch.nn.Module):
    def __init__(self, n:float):
        super().__init__()
        self.n = n
    def forward(self, x):
        return self.n*x

def get_model(dev=None, latent_dim=32, use_norm=True, always_norm=False):
  if use_norm:
    return torch.nn.Sequential(
        direct_norm(torch.nn.Linear(2, latent_dim), kind="two-inf", always_norm=always_norm),
        GroupSort(2),
        direct_norm(torch.nn.Linear(latent_dim, latent_dim), kind="inf", always_norm=always_norm),
        GroupSort(2),
        direct_norm(torch.nn.Linear(latent_dim, latent_dim), kind="inf", always_norm=always_norm),
        GroupSort(2),
        direct_norm(torch.nn.Linear(latent_dim, 1), kind="inf", always_norm=always_norm),
    ).to(dev or "cpu")
  else:
    return torch.nn.Sequential(
      torch.nn.Linear(2, latent_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(latent_dim, latent_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(latent_dim, latent_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(latent_dim, 1),
    ) 

