# %%
import torch
import numpy as np
from matplotlib import pyplot as plt
import ot
from scipy.stats import multivariate_normal as mn
from itertools import product
from tqdm import tqdm
plt.style.use('../paper-dark')
torch.random.manual_seed(1)
np.random.seed(1)

PLOT = False

device = "cuda:0"
dim = 2
n = 50
n_targets = 10

# generate data
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
bins = np.array(list(product(x, y)))
p = mn.pdf(bins, np.zeros(dim), np.eye(dim) * 0.2)
p /= p.sum()
# %%
a = np.ones(n_targets)  # + np.random.randn(n_targets)
b = np.zeros((dim, n_targets))  # np.random.randint(0, 1, (dim, n_targets))
qbins = bins[..., None] * a + b  # np.arange(n_targets) + 1
qs = np.zeros((n**dim, n_targets))
for i in range(n_targets):
    q = p.copy()
    np.random.shuffle(q)
    # q += np.random.randn(*q.shape) * 1
    q -= q.min()
    q /= q.sum()
    qs[:, i] = q
    if PLOT:
        X = qbins[:, 0, i]
        X = X.reshape((n, n))
        Y = qbins[:, 1, i]
        Y = Y.reshape((n, n))
        C = q.reshape((n, n))
        plt.pcolormesh(X, Y, C)
        plt.show()

# Plot OG distrubution
if PLOT:
    X = bins[:, 0]
    X = X.reshape((n, n))
    Y = bins[:, 1]
    Y = Y.reshape((n, n))
    C = p.reshape((n, n))
    plt.pcolormesh(X, Y, C)
    plt.show()
# %%
distances = []
for i in range(n_targets):
    M = ot.dist(bins, qbins[:, :, i], metric="euclidean")
    distances.append(ot.emd2(p, np.ascontiguousarray(qs[:, i]), M))

# %%
# ------------------ KR EMD ------------------
from monotonenorm import direct_norm, GroupSort


class Clamp(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super(Clamp, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, input):
        return torch.clamp(input, self.min_value, self.max_value)


class TimesN(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x * self.n


model = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(2, 128, bias=True), kind="two-inf", always_norm=False),
    GroupSort(2),
    direct_norm(torch.nn.Linear(128, 128, bias=True), kind="inf", always_norm=False),
    GroupSort(2),
    direct_norm(torch.nn.Linear(128, 128, bias=True), kind="inf", always_norm=False),
    GroupSort(2),
    direct_norm(torch.nn.Linear(128, 1, bias=False), kind="inf", always_norm=False),
).to(device)


EPOCHS = 1000
lr_init = 2e-2
lr_final = 1e-4
gamma = (lr_final / lr_init)**(1 / EPOCHS)

bins_tensor = torch.from_numpy(bins).float().to(device)
p_tensor = torch.from_numpy(p).float().view(-1, 1).to(device)

for i, d_emd in enumerate(distances):
    max_emd = 0
    optim = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    qbins_tensor = torch.from_numpy(qbins[:, :, i]).float().to(device)
    # q_tensor = q_tensor.view(-1, 1)
    q_tensor = torch.from_numpy(qs[:, i]).float().view(-1, 1).to(device)
    bar = tqdm(range(EPOCHS))
    for epoch in bar:
        E0 = (model(bins_tensor) * p_tensor).sum()
        E1 = (model(qbins_tensor) * q_tensor).sum()
        emd = E0 - E1
        loss = -emd
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
        if emd.item() > max_emd:
            max_emd = emd.item()
            delta = (max_emd - d_emd) / d_emd * 100
            message = f"{max_emd:.3f} vs {d_emd:.3f} - Delta: {delta:.2f}% @ {epoch}"
            bar.set_description(message)
    print("finished working on tensor: ", i)

# %%
