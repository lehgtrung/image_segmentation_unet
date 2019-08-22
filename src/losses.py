
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import extract_narrow_band
from torch.autograd import Variable


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


def get_length(phi, device):
    conv_filter_x = torch.tensor([[[[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]]]], dtype=torch.float32).to(device)
    conv_filter_y = torch.tensor([[[[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]]]], dtype=torch.float32).to(device)
    grad_x = F.conv2d(phi, conv_filter_x, padding=1)
    grad_y = F.conv2d(phi, conv_filter_y, padding=1)

    grad = (grad_x.pow(2) + grad_y.pow(2) + 1e-5).sqrt()
    dd = delta_dirac(phi - 0.5)
    return torch.mul(grad, dd).sum(-1).sum(-1)


def heaviside1(x, eps=1, pi=np.pi):
    res = x.clone()
    res[x > eps] = 1.0
    res[x < -eps] = 0.0
    a = x[(x >= -eps) & (x <= eps)]
    res[(x >= -eps) & (x <= eps)] = (torch.sin(pi * a/eps)*(1/pi) + a/pi + 1)*0.5
    return res


def heaviside2(x, eps=1e-5, pi=np.pi):
    return 1/2 * (1 + 2/pi * torch.atan(x.div(eps)))


def delta_dirac(x, eps=1, pi=np.pi):
    return eps / (pi * (eps**2 + x.pow(2)))


class ContourLoss(nn.Module):
    def __init__(self, device, mu, normed, withlen):
        self.normed = normed
        self.withlen = withlen
        self.device = device
        self.mu = mu
        super(ContourLoss, self).__init__()

    def forward(self, preds, targets):
        c1 = 1.0
        c2 = 0.0
        eps = 1e-7
        force_inside = (preds - c1).pow(2).mul(targets).sum(-1).sum(-1)
        force_outside = (preds - c2).pow(2).mul(1.0 - targets).sum(-1).sum(-1)
        if self.normed:
            force_inside = (force_inside + eps)/(targets.sum(-1).sum(-1) + eps)
            force_outside = (force_outside + eps)/((1 - targets).sum(-1).sum(-1) + eps)
        if self.withlen:
            contour_len = self.mu * get_length(preds, self.device)
            force = contour_len + force_inside + force_outside
        else:
            force = force_inside + force_outside
        return torch.mean(force)


class ContourLossVer2(nn.Module):
    def __init__(self, device, mu, normed, withlen):
        self.normed = normed
        self.withlen = withlen
        self.device = device
        self.mu = mu
        super(ContourLossVer2, self).__init__()

    def forward(self, preds, targets, nb_mask):
        c1 = 1.0
        c2 = 0.0
        eps = 1e-7
        force_inside = (preds - c1).pow(2).mul(targets).mul(nb_mask).sum(-1).sum(-1)
        force_outside = (preds - c2).pow(2).mul(1.0 - targets).mul(1- nb_mask).sum(-1).sum(-1)
        if self.normed:
            force_inside = (force_inside + eps)/(targets.mul(nb_mask).sum(-1).sum(-1) + eps)
            force_outside = (force_outside + eps)/((1.0 - targets).mul(1 - nb_mask).sum(-1).sum(-1) + eps)
        if self.withlen:
            contour_len = self.mu * get_length(preds, self.device)
            force = contour_len + force_inside + force_outside
        else:
            force = force_inside + force_outside
        return torch.mean(force)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


