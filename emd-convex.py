# %%
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import ot
from tqdm import tqdm
from emd_loss import emd_loss

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

torch.random.manual_seed(1)
np.random.seed(1)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dE = 0

n_tracks_p = 5
n_tracks_q = 5

ps = np.random.randn(n_tracks_p, 2)
pEs = np.random.rand(n_tracks_p)
pEs = pEs / pEs.sum()

qs = np.random.randn(n_tracks_q, 2)
qEs = np.random.rand(n_tracks_q)
qEs = qEs / qEs.sum() * (1 + dE)

pEs_cheat = np.concatenate((pEs, np.array([dE])))

M = ot.dist(ps, qs, metric="euclidean")
M = np.vstack((M, np.ones(M.shape[1])))
distance = ot.emd2(pEs_cheat, qEs, M)
print("EMD:", distance)
# %%

pst = torch.from_numpy(ps).float().to(dev)
qst = torch.from_numpy(qs).float().to(dev)
pEst = torch.from_numpy(pEs).float().to(dev).view(-1, 1)
qEst = torch.from_numpy(qEs).float().to(dev).view(-1, 1)

pst = torch.hstack((pst, pEst))
qst = torch.hstack((qst, qEst))
# %%

emd = emd_loss(pst, qst, device=dev)
print("EMD convex:", emd)


# %%
