import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import ot
from tqdm import tqdm
from copy import deepcopy
from model import get_model

if not os.path.exists("models"):
    os.mkdir("models")

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

torch.random.manual_seed(1)
np.random.seed(1)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", dev)
model = get_model(dev)

dim = 2
size = 1000
n = 20
n_targets = 10

dE = 0

n_tracks_p = 20
n_tracks_q = 15

ps = np.random.randn(n_tracks_p, 2)
pEs = np.random.rand(n_tracks_p)
pEs = pEs / pEs.sum()
np.savez(f"ps.npz", ps=ps, pEs=pEs)

for i in range(10):
    qs = np.random.randn(n_tracks_q, 2)
    qEs = np.random.rand(n_tracks_q)
    qEs = qEs / qEs.sum() * (1 + dE)

    np.savez(f"qs-{i}.npz", qs=qs, qEs=qEs)
    pEs_cheat = np.concatenate((pEs, np.array([dE])))

    M = ot.dist(ps, qs, metric="euclidean")
    M = np.vstack((M, np.ones(M.shape[1])))
    distance = ot.emd2(pEs_cheat, qEs, M)

    EPOCHS = 20000
    lr_init = 2e-2
    lr_final = 1e-4
    gamma = (lr_final / lr_init) ** (1 / EPOCHS)

    pst = torch.from_numpy(ps).float().to(dev)
    qst = torch.from_numpy(qs).float().to(dev)
    pEst = torch.from_numpy(pEs).float().to(dev).view(-1, 1)
    qEst = torch.from_numpy(qEs).float().to(dev).view(-1, 1)

    max_emd = 0
    optim = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    bar = tqdm(range(EPOCHS))
    for epoch in bar:
        mp = model(pst)
        mq = model(qst)
        E0 = (mp * pEst).sum()
        E1 = (mq * qEst).sum()
        emd = E1 - E0 - dE * (mq.max() - 1)
        if emd.item() > max_emd:
            sd = deepcopy(model.state_dict())
            max_emd = emd.item()
            delta = (max_emd - distance) / distance * 100
            message = f"{max_emd:.3f} vs {distance:.3f} - Delta: {delta:.2f}% @ {epoch}"
            bar.set_description(message)
        # update model
        loss = -emd
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
    torch.save(sd, f"models/emd-kr-toy-{i}.pt")
