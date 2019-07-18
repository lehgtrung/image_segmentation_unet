
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets):
        smooth = 1
        batch_size = targets.size(0)
        probs = torch.sigmoid(preds)
        m1 = probs.view(batch_size, -1)
        m2 = targets.view(batch_size, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / batch_size
        return score


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, preds, targets):
        probs = torch.sigmoid(preds)
        m1 = probs.view(-1)
        m2 = probs.view(-1)
        return F.binary_cross_entropy(m1, m2)


def get_length(phi, eps=1e-5):
    grad_x = phi[:, :, 1:, :] - phi[:, :, :-1, :]
    grad_y = phi[:, :, :, 1:] - phi[:, :, :, :-1]

    grad_x = grad_x[:, :, 1:, :-2]**2
    grad_y = grad_y[:, :, :-2, 1:]**2
    grad = grad_x + grad_y
    grad = torch.sqrt(grad + eps).mean(dim=-1).mean(dim=-1)
    return grad


def heaviside(x, eps=1e-5, pi=np.pi):
    res = x.clone()
    res[x > eps] = 1.0
    res[x < -eps] = 0.0
    a = x[(x >= -eps) & (x <= eps)]
    res[(x >= -eps) & (x <= eps)] = (torch.sin(pi * a/eps)*(1/pi) + a/pi + 1)*0.5
    return res


class ContourLoss(nn.Module):
    def __init__(self):
        super(ContourLoss, self).__init__()

    def forward(self, preds, targets):
        probs = torch.sigmoid(preds)
        c1 = torch.tensor([[[1.0]]], dtype=torch.float32)
        c2 = torch.tensor([[[0.0]]], dtype=torch.float32)
        if torch.cuda.is_available():
            c1 = c1.cuda()
            c2 = c2.cuda()
        contour_len = get_length(probs, targets)
        force_inside = (probs - c1.expand_as(probs)).pow(2).mul(targets).mean(-1).mean(-1)
        force_outside = (probs - c2.expand_as(probs)).pow(2).mul(1.0 - targets).mean(-1).mean(-1)
        force = contour_len + force_inside + force_outside
        return torch.mean(force)


