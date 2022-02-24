import torch
import numpy as np
import ot
from tqdm import tqdm

import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})


from toymodel import get_model

plot = False
plotfreq = 1000
save = False#not plot

unconstrained = False
L = 1
EPOCHS = 5000
lr_init = 5e-2
lr_final = 5e-3
gamma = (lr_final / lr_init) ** (1 / EPOCHS)
dev = torch.device("cpu")

nps = 5

ps = np.linspace(0, nps - 1, nps)
qs = ps[np.round(ps) % 2 == 0].reshape(-1, 1)
ps = ps[np.round(ps) % 2 == 1].reshape(-1, 1)
# add some more data point to the immediate left / right does not help
# ps = np.concatenate([ps+.1, ps - .1, ps])
# qs = np.concatenate([qs+.1, qs - .1, qs])
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


def evaluate(seed):
    torch.random.manual_seed(seed)

    model = get_model(unconstrained=unconstrained, dev=dev, L=L, project=False)

    max_emd = 0
    optim = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)

    if plot:
        minx = min(qs.min(), ps.min())
        maxx = max(qs.max(), ps.max())

        _, ax = plt.subplots()

        title = ax.text(0.05, 0.8, "", transform=ax.transAxes)

        nbinsPP = 2
        nbins = (len(ps) + len(qs)) * nbinsPP - nbinsPP + 1

        X = np.linspace(minx, maxx, nbins).reshape(-1, 1)
        Xt = torch.from_numpy(X).float().to(dev)

        ax.scatter(qs[:, 0], qs[:, 0] * 0, s=qEs * 1000, c="royalblue")
        ax.scatter(ps[:, 0], ps[:, 0] * 0, s=pEs * 1000, c="crimson")

        maxys = [-10, 10]

        ax.hlines(maxys, minx, maxx, linestyles="dashed")
        with torch.no_grad():
            h = ax.plot(X, np.random.choice(maxys, len(X)), c="black")

    bar = tqdm(range(EPOCHS))
    for i in bar:
        mp = model(pst)
        mq = model(qst)
        E0 = (mp * pEst).sum()
        E1 = (mq * qEst).sum()
        if unconstrained:
            factor = ((mp - mq.view(-1)).abs() / Mt).max()
            emd = (E1 - E0) / factor
        else:
            emd = E1 - E0
        with torch.no_grad():
            delta = (max_emd - distance) / distance * 100
            if i % 100 == 0:
                if emd > max_emd:
                    max_emd = emd
                    bar.set_description(
                        f"{emd:.3f} vs {distance:.3f} - Delta: {delta:.4f}% @ {i}"
                    )
            if plot and i % plotfreq == 0:
                Yt = model(Xt)
                Yt = Yt - Yt.mean()
                h[0].set_ydata(Yt.numpy())
                plt.pause(1e-10)
                title.set_text(
                    f"{max_emd:.3f} vs {distance:.3f} - Delta: {delta:.4f}% @ {i}"
                )
        loss = -emd
        loss.backward()
        optim.step()
        scheduler.step()
        optim.zero_grad()
    
    if save:
      metric = "good" if delta > -1 else "bad"
      torch.save(model.state_dict(), f"models/{metric}/model{seed}.pt")

    if plot:
      plt.show()
    
    return delta

if __name__ == __name__:
    import multiprocessing as mp
    #evaluate(3)
    results = list(mp.Pool(110).map(evaluate, range(1000)))
    print((np.array(results) > -5).mean())
    breakpoint()
    