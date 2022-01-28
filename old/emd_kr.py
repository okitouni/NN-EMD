# %%
import ot
import energyflow as ef
import torch
from monotonenorm import direct_norm, GroupSort, project_norm
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

device = "cpu"
PLOT = False
# ------------------ KR EMD ------------------

model = torch.nn.Sequential(
    direct_norm(torch.nn.Linear(2, 32), kind="two-inf", always_norm=False),
    GroupSort(2),
    direct_norm(torch.nn.Linear(32, 32), kind="inf", always_norm=False),
    GroupSort(2),
    direct_norm(torch.nn.Linear(32, 1, bias=False), kind="inf", always_norm=False),
).to(device)


# ------------------ Load Data ------------------
# load quark and gluon jets
X, y = ef.qg_jets.load(2000, pad=False)
num = 750

# the jet radius for these jets
R = 0.4

# process jets
Gs, Qs = [], []
for arr, events in [(Gs, X[y == 0]), (Qs, X[y == 1])]:
    for i, x in enumerate(events):
        if i >= num:
            break

        # ignore padded particles and removed particle id information
        x = x[x[:, 0] > 0, :3]

        # center jet according to pt-centroid
        yphi_avg = np.average(x[:, 1:3], weights=x[:, 0], axis=0)
        x[:, 1:3] -= yphi_avg

        # mask out any particles farther than R=0.4 away from center (rare)
        x = x[np.linalg.norm(x[:, 1:3], axis=1) <= R]

        # add to list
        arr.append(x)


# ------------------ BIG LOOP OVER EVENTS ------------------
    for ii in range(10):
        print(f"event{ii} and 0")
        # ------------------ Primal EMD ------------------
        # choose interesting events
        ev0, ev1 = Gs[ii], Gs[0]

        # bin positions
        x0 = ev0[:, 1:]
        x1 = ev1[:, 1:]

        p0_array = ev0[:, 0]
        p1_array = ev1[:, 0]

        p1sum = np.sum(p1_array)
        p0sum = np.sum(p0_array)

        pt_diff = p1sum - p0sum

        # if pt_diff >= 0:
        #     p0 = np.concatenate((p0, np.array([pt_diff])))
        #     x0 = np.vstack((x0, np.zeros(x0.shape[1])))
        # else:
        #     p1 = np.concatenate((p1, np.array([-pt_diff])))
        #     x1 = np.vstack((x1, np.zeros(x1.shape[1])))

        rescale = max(p1sum, p0sum)

        M = ot.dist(x0, x1, 'euclidean')
        # M /= R

        # if pt_diff >= 0:
        #     p0 = np.concatenate((p0_array, np.array([pt_diff])))
        #     p1 = p1_array
        #     M = np.vstack((M, np.ones(M.shape[1])))
        # else:
        #     p1 = np.concatenate((p1_array, np.array([-pt_diff])))
        #     p0 = p0_array
        #     M = np.hstack((M, np.ones(M.shape[0])))

        # d_emd = ot.emd2(p0 / rescale, p1 / rescale, M)  # direct computation of EMD
        # print(d_emd * rescale)

        d_emd = ot.emd2(p0_array / p0sum, p1_array / p1sum,
                        M)  # direct computation of EMD
        print("primal EMD", d_emd)
        # %%
        if PLOT:
            for x, p in zip([x0, x1], [p0_array, p1_array]):
                X = x[:, 0]
                Y = x[:, 1]
                C = p / np.sum(p)
                plt.scatter(X, Y, C * 1000)
            plt.show()
        # ------------------ Dual EMD training ------------------
        # %%
        EPOCHS = 10000
        # lr_init = 1e-3
        # print("lr_init", lr_init)
        lr_init = 2e-2
        lr_final = 1e-4
        gamma = (lr_final / lr_init)**(1 / EPOCHS)
        # lr_final = 1e-5
        # gamma = (lr_final / lr_init)**(1 / EPOCHS)
        optim = torch.optim.SGD(model.parameters(), lr=lr_init)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=gamma)
        # scheduler = torch.optim. lr_scheduler.OneCycleLR(
        #     optim, max_lr=lr_init, epochs=EPOCHS, steps_per_epoch=1, pct_start=0.3)

        x0_tensor = torch.from_numpy(x0).float().to(device)
        x1_tensor = torch.from_numpy(x1).float().to(device)
        p0_tensor = torch.from_numpy(p0_array).float().to(device)
        p1_tensor = torch.from_numpy(p1_array).float().to(device)

        p0_tensor = p0_tensor.view(-1, 1) / p0_tensor.sum()
        p1_tensor = p1_tensor.view(-1, 1) / p1_tensor.sum()

        # %%
        max_emd = 0
        delta = 0
        pbar = tqdm(range(EPOCHS))
        if ii == 0:
            continue
        for epoch in pbar:
            E0 = (model(x0_tensor) * p0_tensor).sum()
            E1 = (model(x1_tensor) * p1_tensor).sum()
            emd = E0 - E1
            loss = -emd
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()
            if emd.item() > max_emd:
                max_emd = emd.item()
                delta = (emd.item() - d_emd) / d_emd * 100
                message = f"{max_emd:.3f} vs {d_emd:.3f} - Delta: {delta:.2f}% @ {epoch}"
                pbar.set_description(message)
        print(f"finsihed. Dual EMD: {max_emd:.3f}, delta:{delta:.2f}% at epoch {epoch}")
        # %%
