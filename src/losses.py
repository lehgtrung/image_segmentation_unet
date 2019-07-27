
import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        preds = preds.squeeze(1)
        targets = targets.squeeze(1)
        smooth = 1
        num = targets.size(0)
        m1 = preds.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class BinaryCrossEntropyLoss2d(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, preds, targets):
        probs_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


def get_length(phi):
    grad_x = phi[:, :, 1:, :] - phi[:, :, :-1, :]
    grad_y = phi[:, :, :, 1:] - phi[:, :, :, :-1]

    grad_x = grad_x[:, :, 1:, :-2]**2
    grad_y = grad_y[:, :, :-2, 1:]**2
    grad = grad_x + grad_y
    grad = torch.sqrt(grad + 1e-5).sum(dim=-1).sum(dim=-1)
    return grad


def heaviside(x, eps=1e-5, pi=np.pi):
    res = x.clone()
    res[x > eps] = 1.0
    res[x < -eps] = 0.0
    a = x[(x >= -eps) & (x <= eps)]
    res[(x >= -eps) & (x <= eps)] = (torch.sin(pi * a/eps)*(1/pi) + a/pi + 1)*0.5
    return res


class ContourLoss(nn.Module):
    def __init__(self, withlen=True):
        self.withlen = withlen
        super(ContourLoss, self).__init__()

    def forward(self, preds, targets):
        # c1 = (probs.mul(targets).sum(-1).sum(-1) / n_inside).unsqueeze(1).unsqueeze(1).expand_as(probs)
        # c2 = (probs.mul(1.0 - targets).sum(-1).sum(-1) / n_outside).unsqueeze(1).unsqueeze(1).expand_as(probs)
        c1 = 1.0
        c2 = 0.0
        force_inside = (preds - c1).pow(2).mul(targets).sum(-1).sum(-1)
        force_outside = (preds - c2).pow(2).mul(1.0 - targets).sum(-1).sum(-1)
        if self.withlen:
            contour_len = get_length(preds)
            force = contour_len + force_inside + force_outside
        else:
            force = force_inside + force_outside
        return torch.mean(force)


