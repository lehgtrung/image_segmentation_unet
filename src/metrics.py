
import torch
import numpy as np
import sklearn.metrics as skmetrics


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
    n = 0
    for pair in zip(preds, masks):
        flat_pred = pair[0].reshape(-1)
        flat_mask = pair[1].reshape(-1).astype(int)
        fpr, tpr, thresholds = skmetrics.roc_curve(flat_mask, flat_pred)
        best_cutoff = find_best_cutoff(fpr, tpr, thresholds)
        try:
            score = skmetrics.roc_auc_score(flat_mask, flat_pred)
            if mode_average:
                total_auc += score
            else:
                score_list.append([best_cutoff, score])
        except ValueError:
            n += 1
    if mode_average:
        return total_auc / (batch_size - n)
    return list(zip(*score_list))


def accuracy(batch_size, masks, preds, mode_average=True, cutoff=0.5):
    """ Accuracy between thresholded prediction and binary masks """
    total_accuracy = .0
    score_list = []
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        score = skmetrics.accuracy_score(flat_mask, flat_pred)
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
        score = skmetrics.jaccard_score(flat_mask, flat_pred)
        if mode_average:
            total_jaccard += score
        else:
            score_list.append(score)
    if mode_average:
        return total_jaccard / batch_size
    return score_list


def confusion_matrix(batch_size, masks, preds, mode_average=True, cutoff=0.5):
    total_sens = .0
    total_spec = .0
    total_preci = .0
    total_f1 = .0
    score_list = []
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        tn, fp, fn, tp = skmetrics.confusion_matrix(flat_mask, flat_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        preci = tp / (tp + fp)
        f1 = (2 * preci * sens)/(preci + sens)
        if mode_average:
            total_sens += sens
            total_spec += spec
            total_preci += preci
            total_f1 += f1
        else:
            score_list.append((sens, spec, preci, f1))
    if mode_average:
        return total_sens / batch_size, total_spec / batch_size, total_preci / batch_size, total_f1 / batch_size
    return zip(*score_list)


def precision_recall_auc(batch_size, masks, preds, mode_average=True, cutoff=0.5):
    total_pr_auc = .0
    score_list = []
    for pair in zip(preds, masks):
        flat_pred = (pair[0].reshape(-1) >= cutoff).astype(int)
        flat_mask = pair[1].reshape(-1).astype(int)
        precision, recall, thresholds = skmetrics.precision_recall_curve(flat_mask, flat_pred)
        precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
        recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
        pr_auc = np.trapz(precision, recall)
        if mode_average:
            total_pr_auc += pr_auc
        else:
            score_list.append(pr_auc)
    if mode_average:
        return total_pr_auc / batch_size
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





