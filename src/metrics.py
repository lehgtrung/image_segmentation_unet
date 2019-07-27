
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def standardize_for_metrics(masks, preds):
    assert preds.shape == masks.shape
    batch_size = preds.shape[0]
    if len(preds.shape) == 4:
        preds = preds.squeeze(1)
        masks = masks.squeeze(1)
    if torch.cuda.is_available():
        preds = preds.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
    else:
        preds = preds.numpy()
        masks = masks.numpy()
    return batch_size, masks, preds


def roc_auc(batch_size, masks, preds):
    """ Area under ROC curve between predicted probabilities and binary masks """
    total_auc = .0
    k = 0
    for pair in zip(preds, masks):
        flat_pred = pair[0].reshape(-1)
        flat_mask = pair[1].reshape(-1).astype(int)
        try:
            total_auc += roc_auc_score(flat_mask, flat_pred)
        except ValueError:
            k += 1
            continue
    if k == batch_size:
        return 0
    return total_auc / (batch_size - k)


def accuracy(batch_size, masks, preds, cutoff=0.5):
    """ Accuracy between thresholded prediction and binary masks """
    total_accuracy = .0
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        total_accuracy += accuracy_score(flat_mask, flat_pred)
    return total_accuracy / batch_size


def jaccard(batch_size, masks, preds, cutoff=0.5):
    """ Intersection over union between thresholded prediction and binary masks """
    total_ji = .0
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        intersection = np.logical_and(flat_mask, flat_pred)
        union = np.logical_or(flat_mask, flat_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        total_ji += iou_score
    return total_ji / batch_size

