import torch
from matplotlib import pyplot as plt
from model import get_model
import ot
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation


np.random.seed(0)
torch.manual_seed(0)


N = 8
points = np.random.rand(N * 2)
ps = points[::2].reshape(-1, 2)
qs = points[1::2].reshape(-1, 2)

Eps = np.ones(len(ps)) / len(ps)
Eqs = np.ones(len(qs)) / len(qs)

# Plot
fig, ax = plt.subplots()
plt.scatter(ps[:, 0], ps[:, 1], s=Eps * 100, c="crimson")
scatter = plt.scatter(qs[:, 0], qs[:, 1], s=Eqs * 100, c="royalblue")
text = plt.text(0.01, 0.01, "", transform=ax.transAxes)


def emd(p, q, pE, qE):
    M = ot.dist(p, q, metric="euclidean")
    return ot.emd2(pE, qE, M)


emd_true = emd(ps, qs, Eps, Eqs)
print(emd_true)

ps = torch.tensor(ps, dtype=torch.float32)
qs = torch.tensor(qs, dtype=torch.float32, requires_grad=True)
Eps = torch.tensor(Eps, dtype=torch.float32).view(-1, 1)
Eqs = torch.tensor(Eqs, dtype=torch.float32).view(-1, 1)


LR = 1e-3
LRq = 1e0
EPOCHS = 1000
# pbar = tqdm(range(EPOCHS))
model = get_model(size=32)
optim = torch.optim.Adam(model.parameters(), lr=LR)
optim_q = torch.optim.SGD([qs], lr=LRq, momentum=0.01, dampening=0.9)
max_emd = 0


def update(i):
    optim.zero_grad()
    optim_q.zero_grad()
    mp = model(ps)
    mq = model(qs)
    E0 = (mp * Eps).sum()
    E1 = (mq * Eqs).sum()
    loss = E0 - E1
    loss.backward()
    qs.grad = - qs.grad
    optim_q.step()
    emd = -loss.item()
    # if emd > 0:
    max_emd = emd
    delta = (max_emd - emd_true) / emd_true * 100
    message = f"{max_emd:.3f} vs {emd_true:.3f} - Delta: {delta:.2f}% @ {i}"
    # pbar.set_description(message)
    if i % 1 == 0:
        optim.step()
        scatter.set_offsets(qs.detach().numpy())
        text.set_text(message)
    return [scatter, text]


animation = FuncAnimation(fig, update, frames=EPOCHS, repeat=False, blit=True)
plt.tight_layout()
plt.show()
