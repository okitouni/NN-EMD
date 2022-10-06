import ot
import numpy as np
import torch

class EMDLoss(torch.nn.Module):
    def forward(self, p, q, pE=None, qE=None):
        if pE is None:
            pE = torch.ones_like(p) / p.shape[0]
        if qE is None:
            qE = torch.ones_like(q) / q.shape[0]
        return (p * pE).sum() - (q * qE).sum()

def get_emd(p, q, pE, qE):
    assert p.shape[0] == np.prod(pE.shape[0])
    assert q.shape[0] == np.prod(qE.shape[0])
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    if isinstance(pE, torch.Tensor):
        pE = pE.detach().cpu().numpy().flatten()
    if isinstance(qE, torch.Tensor):
        qE = qE.detach().cpu().numpy().flatten()

    M = ot.dist(p, q, metric="euclidean")
    return ot.emd2(pE, qE, M)


def cos_sine(theta):
    if type(theta) == torch.Tensor:
        return torch.concat([torch.cos(theta), torch.sin(theta)], dim=1)
    elif type(theta) == np.ndarray:
        return np.concatenate([np.cos(theta), np.sin(theta)], axis=1)


def hinge_loss(pred, target, margin=1, reduction="mean"):
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError("reduction must be one of ['mean', 'sum', 'none']")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if pred.dim() <= 1:
        raise ValueError("pred must have at least 2 dimensions")

    loss = torch.max(torch.zeros_like(pred), margin - pred * target)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

# gradient reversal layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.lambd
        return grad_output.neg() * lambd, None

