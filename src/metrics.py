
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, jaccard_score, roc_curve


def standardize_for_metrics(masks, preds):
    assert preds.shape == masks.shape
    batch_size = preds.shape[0]
    if len(preds.shape) == 4:
        preds = preds.squeeze(1)
        masks = masks.squeeze(1)
    if isinstance(masks, np.ndarray):
        return batch_size, masks, preds
    if torch.cuda.is_available():
        preds = preds.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
    else:
        preds = preds.numpy()
        masks = masks.numpy()
    return batch_size, masks, preds


def roc_auc(batch_size, masks, preds, mode_average=True):
    """ Area under ROC curve between predicted probabilities and binary masks """
    total_auc = .0
    score_list = []
    for pair in zip(preds, masks):
        flat_pred = pair[0].reshape(-1)
        flat_mask = pair[1].reshape(-1).astype(int)
        fpr, tpr, thresholds = roc_curve(flat_mask, flat_pred)
        best_cutoff = find_best_cutoff(fpr, tpr, thresholds)
        score = roc_auc_score(flat_mask, flat_pred)
        if mode_average:
            total_auc += score
        else:
            score_list.append([best_cutoff, score])
    if mode_average:
        return total_auc / batch_size
    return list(zip(*score_list))


def accuracy(batch_size, masks, preds, mode_average=True, cutoff=0.5):
    """ Accuracy between thresholded prediction and binary masks """
    total_accuracy = .0
    score_list = []
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        score = accuracy_score(flat_mask, flat_pred)
        if mode_average:
            total_accuracy += score
        else:
            score_list.append(score)
    if mode_average:
        return total_accuracy / batch_size
    return score_list


def jaccard(batch_size, masks, preds, mode_average=True, cutoff=0.5):
    """ Intersection over union between thresholded prediction and binary masks """
    total_jaccard = .0
    score_list = []
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        score = jaccard_score(flat_mask, flat_pred)
        if mode_average:
            total_jaccard += score
        else:
            score_list.append(score)
    if mode_average:
        return total_jaccard / batch_size
    return score_list


def find_best_cutoff(fpr, tpr, thresholds):
    idx = 0
    best_value = 999  # arbitarily large number
    for i, (f, t, threshold) in enumerate(zip(fpr, tpr, thresholds)):
        value = f**2 + (t - 1)**2
        if value < best_value:
            idx = i
            best_value = value
    return thresholds[idx]




