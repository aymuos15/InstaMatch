import torch

from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes

def lw_metric(pred_cc, gt_cc, num_gt_instances, metric='dsc'):
    """
    Calculate Lesionwise Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        num_gt_instances (int): Number of ground truth instances.

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: Moawad, A.W., Janas, A., Baid, U., Ramakrishnan, D., Saluja, R., Ashraf, N., Maleki, N., Jekel, L., Yordanov, N., Fehringer, P. and Gkampenis, A., 2024. The Brain Tumor Segmentation-Metastases (BraTS-METS) Challenge 2023: Brain Metastasis Segmentation on Pre-treatment MRI. ArXiv, pp.arXiv-2306.
    """
    # Extract the number of classes from the one-hot encoded predictions
    num_classes = pred_cc.shape[0]
    
    # Initialize counters for metric accumulation and false positives
    total_metric_score = 0.0
    total_fps = 0
    
    # Skip background class (index 0) and process each foreground class
    for cls in range(1, num_classes):  # exclude background
        # Check if either prediction or ground truth is empty for this class
        score, handled = _handle_empty_classes(pred_cc[cls], gt_cc[cls])
        if handled:
            # If there's a prediction but no ground truth, count it as a false positive
            total_fps += 1 if score == 0.0 and pred_cc[cls].sum() > 0 else 0
            continue
        
        # Tensors to track false positive and true positive instance IDs
        cls_fp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_tp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_metric_score = []

        # Evaluate each ground truth instance separately
        for cls_instance in range(1, int(gt_cc[cls].max().item()) + 1):
            # Create binary mask for current ground truth instance
            gt_tmp = (gt_cc[cls] == cls_instance)

            # Find all predicted instance IDs that overlap with this ground truth instance
            intersecting_cc = torch.unique(pred_cc[cls][gt_tmp])
            # Remove background (0) from the list of overlapping instances
            intersecting_cc = intersecting_cc[intersecting_cc != 0]

            # Create binary mask for matched prediction instances
            pred_tmp = torch.zeros_like(pred_cc[cls], dtype=torch.bool)
            if len(intersecting_cc) > 0:
                # Add these instances to the list of true positives
                cls_tp = torch.cat([cls_tp.to(gt_tmp.device), intersecting_cc])
                # Mark locations of all intersecting predicted instances
                pred_tmp[torch.isin(pred_cc[cls], intersecting_cc)] = True

            # Calculate the specified metric for this ground truth instance
            instance_score = METRIC_FUNCS[metric](pred_tmp, gt_tmp)
            cls_metric_score.append(instance_score)
        
        # Identify false positive instances (predicted instances that don't overlap with any ground truth)
        cls_mask = (pred_cc[cls] != 0) & (~torch.isin(pred_cc[cls], cls_tp))
        cls_fp = torch.unique(pred_cc[cls][cls_mask], sorted=True)
        cls_fp = cls_fp[cls_fp != 0]

        # Add false positives to the total count
        total_fps += len(cls_fp)
        # Sum up all instance scores for this class
        total_metric_score += torch.sum(torch.tensor(cls_metric_score))
    
    # Calculate the final score by normalizing by total number of instances (TP + FP)
    # This penalizes both missed detections and false positives
    mean_metric_score = total_metric_score / (num_gt_instances + total_fps)
    return mean_metric_score.item()