import torch

from metrics.tools.connected_components import gpu_connected_components
from metrics.instance.helper import METRIC_FUNCS
from metrics.instance.helper import _handle_empty_classes, _calculate_final_score

def cluster_metric(pred, gt, metric='dsc'):
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    for cls in range(1, num_classes):  # exclude background
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        overlay = pred[cls] + gt[cls]
        overlay[overlay > 0] = 1
        labeled_array, num_features = gpu_connected_components(overlay)

        cls_metric_score = torch.zeros(num_features, device=pred.device)
        for cls_instance in range(1, num_features + 1):
            cluster_mask = (labeled_array == cls_instance)
            pred_cluster = pred[cls] * cluster_mask
            gt_cluster = gt[cls] * cluster_mask
            cls_metric_score[cls_instance - 1] = METRIC_FUNCS[metric](pred_cluster, gt_cluster)

        total_metric_score += torch.mean(cls_metric_score)

    return _calculate_final_score(total_metric_score, num_classes).item()
