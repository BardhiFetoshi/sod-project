import torch


def threshold_predictions(preds, threshold=0.5):
    return (preds > threshold).float()


def compute_iou(preds, targets, eps=1e-7):
    preds = threshold_predictions(preds)

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_precision(preds, targets, eps=1e-7):
    preds = threshold_predictions(preds)

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))

    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean().item()


def compute_recall(preds, targets, eps=1e-7):
    preds = threshold_predictions(preds)

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))

    recall = (tp + eps) / (tp + fn + eps)
    return recall.mean().item()


def compute_f1(preds, targets, eps=1e-7):
    precision = compute_precision(preds, targets)
    recall = compute_recall(preds, targets)

    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return f1


def compute_mae(preds, targets):
    mae = torch.abs(preds - targets).mean()
    return mae.item()
