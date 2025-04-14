import torch

from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes

def lw_metric(pred_cc, gt_cc, num_gt_instances, metric='dsc'):
    num_classes = pred_cc.shape[0]
    
    total_metric_score = 0.0
    total_fps = 0
    
    for cls in range(1, num_classes):  # exclude background
        score, handled = _handle_empty_classes(pred_cc[cls], gt_cc[cls])
        if handled:
            total_fps += 1 if score == 0.0 and pred_cc[cls].sum() > 0 else 0
            continue
        
        cls_fp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_tp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_metric_score = []

        for cls_instance in range(1, int(gt_cc[cls].max().item()) + 1):
            gt_tmp = (gt_cc[cls] == cls_instance)

            intersecting_cc = torch.unique(pred_cc[cls][gt_tmp])
            intersecting_cc = intersecting_cc[intersecting_cc != 0]

            pred_tmp = torch.zeros_like(pred_cc[cls], dtype=torch.bool)
            if len(intersecting_cc) > 0:
                cls_tp = torch.cat([cls_tp.to(gt_tmp.device), intersecting_cc])
                pred_tmp[torch.isin(pred_cc[cls], intersecting_cc)] = True

            # Calculate metric for this instance and add to total
            instance_score = METRIC_FUNCS[metric](pred_tmp, gt_tmp)
            cls_metric_score.append(instance_score)
        
        cls_mask = (pred_cc[cls] != 0) & (~torch.isin(pred_cc[cls], cls_tp))
        cls_fp = torch.unique(pred_cc[cls][cls_mask], sorted=True)
        cls_fp = cls_fp[cls_fp != 0]

        total_fps += len(cls_fp)
        total_metric_score += torch.sum(torch.tensor(cls_metric_score))
    
    mean_metric_score = total_metric_score / (num_gt_instances + total_fps)
    return mean_metric_score.item()