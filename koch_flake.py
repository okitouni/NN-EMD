# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

n = 4


def get_p(p1, p2):
    z = complex(p2[0] - p1[0], p2[1] - p1[1])
    zr = complex(1 / 2, -3**(1 / 2) / 2)
    z = z * zr
    p = [z.real + p1[0], z.imag + p1[1]]
    return p


ir = 10
ip1 = [0, 0]
ip2 = [ir, 0]
ip3 = [ir / 2, ir / 2 * (3**(1 / 2))]


def gps(n):
    if n == 0:
        return [ip1, ip2, ip3]
    else:
        points = gps(n - 1)
        points.append(ip1)
        i = 0
        ls = []
        # print(points)
        while(1):
            p1 = points[i]
            p2 = points[i + 1]
            p11 = [p1[0] + (p2[0] - p1[0]) / 3, p1[1] + (p2[1] - p1[1]) / 3]
            p12 = [p1[0] + (p2[0] - p1[0]) / 3 * 2, p1[1] + (p2[1] - p1[1]) / 3 * 2]
            p = get_p(p11, p12)
            ls.append(p1)
            ls.append(p11)
            ls.append(p)
            ls.append(p12)
            i = i + 1
            if len(points) - 1 < i + 1:
                break
        return ls


ps = gps(n)
ll = []
for p in ps:
    ll.append(p[0])
ll.append(ps[0][0])
hl = []
for p in ps:
    hl.append(p[1])
hl.append(ps[0][1])

left = np.array(ll)
height = np.array(hl)
plt.plot(left, height)
plt.show()
# %%

ps = np.array(ps) - np.mean(ps, axis=0)
ps_list = []
for i in [1, 1.5, 2.5, 4.5, 10]:
    # for i in [1, 2]:
    flake = ps / i
    ps_list.append(flake)
    plt.plot(*flake.T, )
plt.show()
# %%
x_train = torch.tensor(np.vstack(ps_list)).float()
y = torch.ones(len(ps), 1) * torch.arange(len(ps_list))
y = y.T
y = y.float().reshape(-1, 1)
y_train = ((y - y.min()) / (y.max() - y.min()) * 2 - 1)

plt.scatter(*x_train.T, c=y_train.squeeze(), s=10)
plt.colorbar()
plt.show()

# device = "cuda:0"
device = "cpu"
# %%
from monotonenorm import direct_norm, GroupSort
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def no_norm(x, **kwargs):
    return x


model = torch.nn.Sequential(
    no_norm(torch.nn.Linear(2, 512, bias=True), kind="one-inf", always_norm=False),
    torch.nn.ReLU(),
    no_norm(torch.nn.Linear(512, 512, bias=True), kind="inf", always_norm=False),
    torch.nn.ReLU(),
    no_norm(torch.nn.Linear(512, 512, bias=True), kind="inf", always_norm=False),
    torch.nn.ReLU(),
    no_norm(torch.nn.Linear(512, 512, bias=True), kind="inf", always_norm=False),
    torch.nn.ReLU(),
    no_norm(torch.nn.Linear(512, 1, bias=True), kind="inf", always_norm=False),
).to(device)

# model = torch.nn.Sequential(
#     direct_norm(torch.nn.Linear(2, 512, bias=True), kind="one-inf", always_norm=False),
#     GroupSort(2),
#     direct_norm(torch.nn.Linear(512, 512, bias=True), kind="inf", always_norm=False),
#     GroupSort(2),
#     direct_norm(torch.nn.Linear(512, 512, bias=True), kind="inf", always_norm=False),
#     GroupSort(2),
#     direct_norm(torch.nn.Linear(512, 1, bias=True), kind="inf", always_norm=False),
# ).to(device)
# %%
EPOCHS = 1000
BATCH_SIZE = 1024
init_lr = 0.01
final_lr = 0.001

lr_decay = (final_lr / init_lr) ** (1 / (EPOCHS * 0.5))

loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=len(loader), gamma=lr_decay)

pbar = tqdm(range(len(loader) * EPOCHS))
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        breakpoint()
        loss = torch.nn.functional.mse_loss(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(f"epoch: {epoch} loss: {loss.item():.4f}")
        pbar.update(1)

# %%
n_grid = 200
X, Y = np.meshgrid(np.linspace(-6, 6, n_grid), np.linspace(-6, 6, n_grid))
grid = torch.tensor(np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))).float().to(device)
with torch.no_grad():
    y_pred = model(grid)
    y_pred = y_pred.numpy().reshape(n_grid, n_grid)

fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
plt.pcolormesh(X, Y, y_pred, alpha=1)
plt.colorbar()
cs = plt.contour(X, Y, y_pred, colors="k", linewidths=np.linspace(1, .6, len(ps_list)),
                 levels=np.unique(y_train), linestyles="solid")
plt.clabel(cs, inline=True, fontsize=10)
for i in range(len(ps_list)):
    plt.plot(*ps_list[i].T, lw=1, c="red", alpha=0.4)
# plt.scatter(*x_train.T, c=y_train.squeeze(), s=30)
plt.scatter(*x_train.T, c="w", s=30, alpha=0.5)
plt.savefig("output/Unc-Flake.png")
plt.show()
torch.save(model.state_dict(), "output/Unc-Flake.pt")

# %%
