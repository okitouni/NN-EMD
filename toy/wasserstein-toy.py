import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import ot
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12})


from toymodel import get_model

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False

torch.random.manual_seed(3)

plot=True

unconstrained = False
L = 1
EPOCHS = 15000
lr_init = 5e-2
lr_final = 2e-4
gamma = (lr_final / lr_init) ** (1 / EPOCHS)
dev = torch.device('cpu')

model = get_model(unconstrained=unconstrained, dev=dev, L=L)

nps = 11

ps = np.linspace(0,10,nps)#.reshape(-1, 1)
qs = ps[np.round(ps) % 2 == 0].reshape(-1, 1)
ps = ps[np.round(ps) % 2 == 1].reshape(-1, 1)
pEs = np.ones(len(ps))
pEs = pEs / pEs.sum()
np.savez(f"ps.npz", ps=ps, pEs=pEs)

qEs = np.ones(len(qs))
qEs = qEs / qEs.sum()
np.savez(f"qs.npz", qs=qs, qEs=qEs)

M = ot.dist(ps, qs, metric="euclidean")
pdists = torch.tensor(M.min(axis=1).reshape(-1, 1)).to(dev)
qdists = torch.tensor(M.min(axis=0).reshape(-1, 1)).to(dev)
distance = ot.emd2(pEs, qEs, M)
Mt = torch.tensor(M).to(dev)


pst = torch.from_numpy(ps).float().to(dev)
qst = torch.from_numpy(qs).float().to(dev)
pEst = torch.from_numpy(pEs).float().to(dev).view(-1, 1)
qEst = torch.from_numpy(qEs).float().to(dev).view(-1, 1)

max_emd = 0
optim = torch.optim.Adam(model.parameters(), lr=lr_init)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

bar = tqdm(range(EPOCHS))

minx = min(qs.min(), ps.min())
maxx = max(qs.max(), ps.max())

if plot:

  fig, ax = plt.subplots()

  title = ax.text(0.05, 0.8, "", transform=ax.transAxes)

  nbinsPP = 2
  nbins = (len(ps)+ len(qs))*nbinsPP - nbinsPP + 1
  diff_from_0 = ((len(ps) + len(qs)) * nbinsPP * 4) ** -1

  X = np.linspace(minx, maxx, nbins).reshape(-1, 1)
  Xt = torch.from_numpy(X).float().to(dev)

  ax.scatter(qs[:, 0], qs[:, 0] * 0, s=qEs * 1000, c="royalblue")
  ax.scatter(ps[:, 0], ps[:, 0] * 0, s=pEs * 1000, c="crimson")

  maxys = [-10, 10]#[-.25 - diff_from_0, .25 - diff_from_0]

  ax.hlines(maxys, minx, maxx, linestyles="dashed")
  with torch.no_grad():
    h = ax.plot(X, np.random.choice(maxys, len(X)), c="black")

bar = tqdm(range(EPOCHS))
for i in bar:
    with torch.no_grad():
      up = torch.randn_like(pst).to(dev) / 100
      uq = torch.randn_like(qst).to(dev) / 100
    mp = model(pst + up)
    mq = model(qst + uq)
    E0 = (mp * pEst).sum()
    E1 = (mq * qEst).sum()
    if unconstrained:
      factor = ((mp - mq.view(-1)).abs() / Mt).max()
      emd = (E1 - E0) / factor
    else:
      emd = E1 - E0
    with torch.no_grad():
      if i % 10 == 0:
        full_emd = (model(qst) * qEst).sum() - (model(pst) * pEst).sum()
        if full_emd > max_emd:
          delta = (max_emd - distance) / distance * 100
          max_emd = full_emd
          if plot:
            Yt = model(Xt)
            Yt = Yt - Yt.mean()
            h[0].set_ydata(Yt.numpy())
            plt.pause(1e-10)
            title.set_text(f"{max_emd:.3f} vs {distance:.3f} - Delta: {delta:.4f}% @ {i}")
          bar.set_description(f"{max_emd:.3f} vs {distance:.3f} - Delta: {delta:.4f}% @ {i}")
    loss = -emd
    loss.backward()
    optim.step()
    scheduler.step()
    optim.zero_grad()

torch.save(model.state_dict(), f"model.pt")
plt.show()
