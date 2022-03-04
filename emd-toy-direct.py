import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import ot
from tqdm import tqdm
from model import get_model

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

torch.random.manual_seed(3)
np.random.seed(3)


L = 1
#dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev = torch.device('cpu')

dE = 0

n_tracks_p = 3
n_tracks_q = 3

n_models = 1

ps = np.random.uniform(0,1, (n_tracks_p, 2))
pEs = np.ones(n_tracks_p)
pEs = pEs / pEs.sum()
np.savez(f"ps.npz", ps=ps, pEs=pEs)

for i in range(1):
    models = [get_model(dev, L, bjorck=False) for _ in range(n_models)]
    qs = np.random.uniform(0,1, (n_tracks_q, 2))
    qEs = np.ones(n_tracks_q)
    qEs = qEs / qEs.sum() * (1 + dE)

    np.savez(f"qs-{i}.npz", qs=qs, qEs=qEs)
    pEs_cheat = np.concatenate((pEs, np.array([dE])))

    M = ot.dist(ps, qs, metric="euclidean")
    M = np.vstack((M, np.ones(M.shape[1])))
    pdists = torch.tensor(M[:-1].min(axis=1).reshape(-1, 1)).to(dev)
    qdists = torch.tensor(M[:-1].min(axis=0).reshape(-1, 1)).to(dev)
    distance, matrix = ot.emd2(pEs_cheat, qEs, M, return_matrix=True)
    Mt = torch.tensor(M[:-1]).to(dev)
    
    EPOCHS=100000
    lr_init = 1e-7

    TM = torch.ones((n_tracks_p, n_tracks_q), requires_grad=True).to(dev)
    optimizer = torch.optim.SGD([TM], lr=lr_init)

    def penalty(TM: torch.tensor, pEs:torch.tensor, qEs:torch.tensor) -> torch.tensor:
      pen1 = (TM.sum(dim=1) - pEs).norm(p=2)
      pen2 = (TM.sum(dim=0) - qEs).norm(p=2)
      return pen1 + pen2

    pEst = torch.from_numpy(pEs).float().to(dev)
    qEst = torch.from_numpy(qEs).float().to(dev)

    bar = tqdm(range(EPOCHS))
    for epoch in bar:
      pen = penalty(TM, pEst, qEst)
      EMD = (TM * Mt).sum()
      loss = EMD + pen * 1e3
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      delta = (EMD - distance) / distance * 100
      bar.set_description(f"{EMD:.3f} vs {distance:.3f} - Delta: {delta:.2f}%, pen {pen.item():.3e} @ {epoch}")
