import torch

from metrics.instance.helper import METRIC_FUNCS
from metrics.instance.helper import _handle_empty_classes, _calculate_final_score

def blob_metric(pred, gt, metric='dsc'):
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    for cls in range(1, num_classes):
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        unique_cls_labels = torch.unique(gt[cls][gt[cls] != 0])
        cls_metric_score = torch.zeros(unique_cls_labels.shape[0], device=pred.device)

        for i, cls_blob_label in enumerate(unique_cls_labels):
            cls_label_mask = gt[cls] > 0
            cls_label_mask = ~cls_label_mask
            cls_label_mask[gt[cls] == cls_blob_label] = 1

            the_cls_label = gt[cls] == cls_blob_label
            the_cls_label_int = the_cls_label.int()
            masked_output = pred[cls] * cls_label_mask

            cls_metric_score[i] = METRIC_FUNCS[metric](masked_output, the_cls_label_int)

        total_metric_score += torch.mean(cls_metric_score)
    
    return _calculate_final_score(total_metric_score, num_classes).item()