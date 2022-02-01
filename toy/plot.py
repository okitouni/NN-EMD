import matplotlib.pyplot as plt
import numpy as np
import torch
from toymodel import get_model

qs, qEs = np.load(f"qs.npz").values()
ps, pEs = np.load("ps.npz").values()


model = get_model(unconstrained=False)

model.load_state_dict(torch.load(f"model.pt"))

# plt.scatter(ps[:, 0], ps[:, 1], s=pEs * 1000, c="crimson")
# plt.scatter(qs[:, 0], qs[:, 1], s=qEs * 1000, c="royalblue")

if qs.shape[1] == 1:
    minx = min(qs.min(), ps.min())
    maxx = max(qs.max(), ps.max())

    X = np.linspace(minx, maxx, 100).reshape(-1, 1)
    with torch.no_grad():
        Y = model(torch.from_numpy(X).float())
        Y = Y - Y.mean()
        plt.plot(X, Y.cpu().numpy(), c="black")
        plt.scatter(qs[:, 0], qs[:, 0] * 0, s=qEs * 1000, c="royalblue")
        plt.scatter(ps[:, 0], ps[:, 0] * 0, s=pEs * 1000, c="crimson")
        plt.show()


if qs.shape[1] == 2:

    nb = 100

    minx = min(qs[:, 0].min(), ps[:, 0].min())
    maxx = max(qs[:, 0].max(), ps[:, 0].max())
    miny = min(qs[:, 1].min(), ps[:, 1].min())
    maxy = max(qs[:, 1].max(), ps[:, 1].max())

    X, Y = np.meshgrid(np.linspace(minx, maxx, nb), np.linspace(miny, maxy, nb))

    XY = np.vstack([X.flatten(), Y.flatten()]).T
    with torch.no_grad():
        Z = model(torch.tensor(XY).float())

        # 3d plot of the surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            ps[:, 0],
            ps[:, 1],
            model(torch.tensor(ps).float()),
            s=pEs * 1000,
            c="crimson",
        )
        ax.scatter(
            qs[:, 0],
            qs[:, 1],
            model(torch.tensor(qs).float()),
            s=qEs * 1000,
            c="royalblue",
        )
        ax.plot_surface(
            X, Y, Z.detach().numpy().reshape(nb, nb), cmap="viridis", edgecolor="none"
        )

        # plt.pcolormesh(X, Y, Z.numpy().reshape(nb, nb), shading="auto", alpha=0.5)
        # plt.colorbar()
        # plt.scatter(qbins[:, 0], qbins[:, 1], c=q, s=q*100000, alpha=.5)
        # plt.scatter(pbins[:, 0], pbins[:, 1], c=p, s=p*100000, alpha=.5, marker='x')
        plt.show()

