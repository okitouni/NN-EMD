import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from util import get_emd, cos_sine
from matplotlib.animation import FFMpegWriter
from emd_loss import emd_loss
import time

import cProfile


SEED = 0
np.random.seed(SEED)
torch.manual_seed(2)


EPOCHS = 1000
N = 20
N_circ = 5
N_theta = 40
theta = np.linspace(0, 2 * np.pi, N_theta).reshape(-1, 1)
ps = []
for i in range(N_circ):
    ps.append(np.random.rand(N, 2) + np.random.randint(0, 10, size=(2,)) - 5)
ps = np.concatenate(ps, axis=0)

qs = []
for i in range(N_circ):
    loc = np.zeros(2)
    r = 1
    qs.append(r * (cos_sine(theta) + loc))
qs = np.concatenate(qs, axis=0)

# Eps = np.ones(len(ps)) / len(ps)
Eps = np.random.uniform(0, 1, len(ps))
Eps /= sum(Eps)
Eqs = np.random.uniform(0, 1, N_circ)
Eqs /= sum(Eqs)
Eqs = Eqs.repeat(N_theta) / N_theta


emd_true = get_emd(ps, qs, Eps, Eqs)
print(emd_true)


SAVE = True
if SAVE:
    mpl.use("Agg")
LR = 10
LR_f = 1
gamma = (LR_f / LR)**(1 / EPOCHS)
LRq = 1
LRq_f = 5e-2
gammaq = (LRq_f / LRq)**(1 / EPOCHS)
USE_NORM = False
ALWAYS_NORM = False
pbar = tqdm(total=EPOCHS) if SAVE else EPOCHS

# Plot
fig, ax = plt.subplots()
plt.scatter(ps[:, 0], ps[:, 1], s=Eps * 200, c="crimson")
scatter = plt.scatter(qs[:, 0], qs[:, 1], s=Eqs * 100, c="royalblue")
text = plt.text(0.01, 0.01, "", transform=ax.transAxes)

ps = torch.tensor(ps, dtype=torch.float32)
r = torch.randint(1, 2, size=(N_circ, 1), dtype=torch.float32, requires_grad=True)
loc = torch.randn((N_circ, 2), dtype=torch.float32, requires_grad=True)
theta = torch.tensor(theta, dtype=torch.float32)

qs = torch.concat([r * cos_sine(theta) + loc for r, loc in zip(r, loc)])
Eps = torch.tensor(Eps, dtype=torch.float32).view(-1, 1)
# Eqs = torch.tensor(Eqs, dtype=torch.float32).view(-1, 1)
Ei = torch.rand(N_circ, dtype=torch.float32, requires_grad=True)
psE = torch.hstack((ps, Eps))

params = [{'params': r, 'lr': 6e-2, 'dampening': 0.},
          {'params': loc, 'lr': 10},
        #   {'params': Ei, 'lr': 0}
          ]
optim_q = torch.optim.SGD(params, lr=LRq, momentum=0.02, dampening=0.9)

max_emd = 0

targets = torch.vstack([-torch.ones_like(Eps), torch.ones(len(qs), 1)])

def update(i, save=False):
    Eqs = torch.softmax(Ei.repeat_interleave(N_theta), 0).view(-1, 1)
    qs = torch.concat([r * cos_sine(theta) + loc for r, loc in zip(r, loc)])
    optim_q.zero_grad()
    #TODO define loss
    qsE = torch.hstack((qs, Eqs))
    # %%

    loss = emd_loss(psE, qsE)
    loss.backward()
    optim_q.step()
    max_emd = loss.item()
    message = f"{max_emd:.3f} @ {i}"
    if save:
        pbar.set_description(message)
        pbar.update()
    if i % 1 == 0:
        scatter.set_offsets(qs.detach().numpy())
        scatter.set_sizes(Eqs.detach().numpy().flatten() * 100)
        text.set_text(message)
    return [scatter, text]

#animation = cProfile.run("FuncAnimation(fig, update, frames=EPOCHS, repeat=False, blit=True, fargs=(SAVE, ))")
animation = FuncAnimation(fig, update, frames=EPOCHS, repeat=False, blit=True, fargs=(SAVE, ))
plt.tight_layout()

if SAVE:
    timestamp = time.strftime("%m%d%H%M%S")
    # animation.save("joint_train_OC1.mp4", fps=60)
    name = f"animations/{timestamp}Circles{N_circ}_parts{N}.mp4"
    if ALWAYS_NORM:
        name = name.replace(".mp4", "_alwaysnorm.mp4")
    writer = FFMpegWriter(fps=60, metadata=dict(artist="Me"), bitrate=1800)
    animation.save(name, writer=writer)
    print("saved to ", name)
    plt.close(fig)
else:
    plt.show()
