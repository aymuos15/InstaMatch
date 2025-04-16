import torch
from scipy.optimize import linear_sum_assignment

from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes

def create_match_dict(pred_label_cc, gt_label_cc, metric=None):
    """
    Creates a mapping between predicted and ground truth instance labels for panoptic evaluation.
    
    This function constructs bipartite relationships between prediction and ground truth 
    instances that have any spatial overlap, and calculates their similarity scores.
    """
    # Initialize dictionaries to track relationships and scores
    pred_to_gt = {}     # Maps prediction IDs to ground truth IDs they overlap with
    gt_to_pred = {}     # Maps ground truth IDs to prediction IDs they overlap with
    dice_scores = {}    # Stores similarity scores between each pred-gt pair

    # Extract unique instance labels, excluding background (0)
    pred_labels = torch.unique(pred_label_cc)[1:]
    gt_labels = torch.unique(gt_label_cc)[1:]

    # Create binary masks for each instance label
    pred_masks = {label.item(): pred_label_cc == label for label in pred_labels}
    gt_masks = {label.item(): gt_label_cc == label for label in gt_labels}

    # For each prediction-gt pair, check for overlap and calculate similarity
    for pred_item, pred_mask in pred_masks.items():
        for gt_item, gt_mask in gt_masks.items():
            # Check if there's any spatial overlap between this prediction and ground truth
            if torch.any(torch.logical_and(pred_mask, gt_mask)):
                # Record the relationship in both directions
                pred_to_gt.setdefault(pred_item, []).append(gt_item)
                gt_to_pred.setdefault(gt_item, []).append(pred_item)
                # Calculate similarity score for this pair
                dice_scores[(pred_item, gt_item)] = metric(pred_mask, gt_mask)

    # Ensure all instances have entries in the mapping dictionaries, even if they don't match
    for gt_item in gt_labels:
        gt_to_pred.setdefault(gt_item.item(), [])
    for pred_item in pred_labels:
        pred_to_gt.setdefault(pred_item.item(), [])

    return {"pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred, "dice_scores": dice_scores}

def get_all_matches(matches):
    match_data = []

    for gt, preds in matches["gt_to_pred"].items():
        if not preds:
            match_data.append((None, gt, 0.0))
        else:
            for pred in preds:
                dice_score = matches["dice_scores"].get((pred, gt), 0.0)
                match_data.append((pred, gt, dice_score))

    for pred, gts in matches["pred_to_gt"].items():
        if not gts:
            match_data.append((pred, None, 0.0))

    return match_data

def optimal_matching(match_data):
    predictions = set()
    ground_truths = set()
    valid_matches = []

    for pred, gt, score in match_data:
        if pred is not None and gt is not None:
            predictions.add(pred)
            ground_truths.add(gt)
            valid_matches.append((pred, gt, score))

    pred_to_index = {pred: i for i, pred in enumerate(predictions)}
    gt_to_index = {gt: i for i, gt in enumerate(ground_truths)}

    cost_matrix = torch.ones((len(predictions), len(ground_truths)))

    for pred, gt, score in valid_matches:
        i, j = pred_to_index[pred], gt_to_index[gt]
        cost_matrix[i, j] = 1 - score

    #todo: Use a torch variant here?
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

    optimal_matches = []
    for i, j in zip(row_ind, col_ind):
        pred = list(predictions)[i]
        gt = list(ground_truths)[j]
        score = 1 - cost_matrix[i, j].item()
        optimal_matches.append((pred, gt, score))

    return optimal_matches

def pq_metric(pred_cc, gt_cc, metric='dsc'):
    """
    Calculate Panoptic Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: Kirillov, A., He, K., Girshick, R., Rother, C. and Dollár, P., 2019. Panoptic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 9404-9413).
    """
    # Extract the number of classes from prediction tensor
    num_classes = pred_cc.shape[0]
    
    # Initialize counters for tracking overall performance
    total_metric_score = 0.0  # Sum of quality scores for matched instances
    total_tps = 0  # True positives (matched instances)
    total_fps = 0  # False positives (unmatched predictions)
    total_fns = 0  # False negatives (unmatched ground truths)
    
    # Process each class separately, skipping background (class 0)
    for cls in range(1, num_classes):
        # Handle special cases (empty predictions or ground truths)
        score, handled = _handle_empty_classes(pred_cc[cls], gt_cc[cls])
        if handled:
            # Update counters based on special case outcomes:
            # - Perfect match counts as a true positive
            # - Prediction without ground truth counts as false positive
            # - Ground truth without prediction counts as false negative
            total_tps += 1 if score == 1.0 else 0
            total_fps += 1 if score == 0.0 and pred_cc[cls].sum() > 0 else 0
            total_fns += 1 if score == 0.0 and gt_cc[cls].sum() > 0 else 0
            continue

        # Initialize class-level counters
        cls_tp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_fp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_fn = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_metric_score = 0.0

        # Create bipartite relationships between predictions and ground truths
        matches = create_match_dict(pred_cc[cls], gt_cc[cls], metric=METRIC_FUNCS[metric])
        match_data = get_all_matches(matches)

        # Count false positives (predictions with no matching ground truth)
        cls_fp = sum(1 for pred, gt, _ in match_data if gt is None)
        # Count false negatives (ground truths with no matching prediction)
        cls_fn = sum(1 for pred, gt, _ in match_data if pred is None)

        # Find optimal one-to-one assignment between predictions and ground truths
        # that maximizes total similarity score
        optimal_matches = optimal_matching(match_data)

        # Number of true positives equals number of matched pairs
        cls_tp = len(optimal_matches)
        
        # If no true positives for this class, metric is 0 (complete failure)
        if cls_tp == 0:
            return 0.0
        
        # Sum up the quality scores for all matched pairs in this class
        cls_metric_score = sum(score for _, _, score in optimal_matches)

        # Update global counters with class-specific results
        total_tps += cls_tp
        total_fps += cls_fp
        total_fns += cls_fn
        total_metric_score += cls_metric_score

    # Calculate final panoptic quality score
    # This formula balances both recognition quality (TP vs FP/FN)
    # and segmentation quality (similarity scores of matches)
    mean_metric_score = total_metric_score / (total_fns + total_tps + total_fps)

    return mean_metric_score