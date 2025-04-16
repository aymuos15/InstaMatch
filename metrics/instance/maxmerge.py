import torch
from scipy.optimize import linear_sum_assignment

from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes

def create_match_dict(pred_label_cc, gt_label_cc, metric=None):
    """Creates a mapping between predicted and ground truth instance labels."""
    # Dictionaries to track instance relationships and matching scores
    pred_to_gt = {}
    gt_to_pred = {}
    dice_scores = {}

    # Get unique instance labels from both prediction and ground truth
    pred_labels = torch.unique(pred_label_cc)
    gt_labels = torch.unique(gt_label_cc)

    # Filter out background (label 0) from both prediction and ground truth
    pred_labels = pred_labels[pred_labels != 0]
    gt_labels = gt_labels[gt_labels != 0]

    # Create binary masks for each instance label
    pred_masks = {label.item(): (pred_label_cc == label) for label in pred_labels}
    gt_masks = {label.item(): (gt_label_cc == label) for label in gt_labels}

    # Find all intersecting pairs between prediction and ground truth instances
    for pred_item, pred_mask in pred_masks.items():
        for gt_item, gt_mask in gt_masks.items():
            # Check if there's any overlap between this prediction and ground truth
            intersection = torch.logical_and(pred_mask, gt_mask)
            if torch.any(intersection):
                # Record the relationship in both directions
                pred_to_gt.setdefault(pred_item, []).append(gt_item)
                gt_to_pred.setdefault(gt_item, []).append(pred_item)
                # Calculate matching score for this pair
                dice_scores[(pred_item, gt_item)] = metric(pred_mask, gt_mask)

    # Handle over-segmentation by computing merged prediction scores
    # (when multiple predictions match a single ground truth)
    for gt_item, pred_items in gt_to_pred.items():
        if len(pred_items) > 1:
            # Create a mask combining all predicted instances that match this ground truth
            combined_pred_mask = torch.zeros_like(pred_label_cc, dtype=torch.bool)
            gt_mask = gt_label_cc == gt_item
            for pred_item in pred_items:
                combined_pred_mask |= pred_label_cc == pred_item
            # Calculate score for the merged prediction
            dice_scores[('many', gt_item)] = metric(combined_pred_mask, gt_mask)

    # Ensure all instances have entries in the mapping dictionaries, even if they don't match
    for gt_item in gt_labels:
        gt_to_pred.setdefault(gt_item.item(), [])
    for pred_item in pred_labels:
        pred_to_gt.setdefault(pred_item.item(), [])

    return {
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
        "dice_scores": dice_scores
    }


def get_all_matches(matches):
    """
    Processes the matching dictionary to extract all possible matches between predictions and ground truths,
    including special handling for merged prediction cases.
    """
    match_data = []

    # First, process ground truth instances and their potential matches
    for gt, preds in matches["gt_to_pred"].items():
        # Check if we have a merged prediction score for this ground truth
        many_key = ("many", gt)
        if many_key in matches["dice_scores"]:
            # Get the merged prediction score
            many_score = matches["dice_scores"][many_key]
            # Get individual prediction scores for comparison
            individual_scores = [matches["dice_scores"].get((pred, gt), 0.0) for pred in preds]
            # If merged score is better than any individual score, use it instead
            if many_score > max(individual_scores, default=0.0):
                match_data.append(("many", gt, many_score))
                continue  # Skip individual matches if merged prediction is better

        # Handle unmatched ground truth instances (false negatives)
        if not preds:
            match_data.append((None, gt, 0.0))
        else:
            # Add all individual prediction-to-gt matches
            for pred in preds:
                score = matches["dice_scores"].get((pred, gt), 0.0)
                match_data.append((pred, gt, score))

    # Handle unmatched predictions (false positives)
    for pred, gts in matches["pred_to_gt"].items():
        if not gts:
            match_data.append((pred, None, 0.0))

    return match_data


def optimal_matching(match_data):
    """
    Finds the optimal one-to-one assignment between predictions and ground truths
    using the Hungarian algorithm to maximize the total matching score.
    """
    # Extract unique prediction and ground truth IDs that have valid matches
    predictions = set()
    ground_truths = set()
    valid_matches = []

    for pred, gt, score in match_data:
        if pred is not None and gt is not None:
            predictions.add(pred)
            ground_truths.add(gt)
            valid_matches.append((pred, gt, score))

    # Create mapping from IDs to indices for the cost matrix
    pred_to_index = {pred: i for i, pred in enumerate(predictions)}
    gt_to_index = {gt: i for i, gt in enumerate(ground_truths)}

    # Initialize cost matrix with ones (worst possible cost)
    cost_matrix = torch.ones((len(predictions), len(ground_truths)))

    # Fill in the cost matrix with actual matching costs (1 - score, since Hungarian algorithm minimizes cost)
    for pred, gt, score in valid_matches:
        i, j = pred_to_index[pred], gt_to_index[gt]
        cost_matrix[i, j] = 1 - score

    # Run Hungarian algorithm to find optimal assignment that minimizes total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

    # Convert optimal assignment back to prediction and ground truth IDs with scores
    optimal_matches = []
    for i, j in zip(row_ind, col_ind):
        pred = list(predictions)[i]
        gt = list(ground_truths)[j]
        score = 1 - cost_matrix[i, j].item()  # Convert cost back to score
        optimal_matches.append((pred, gt, score))

    return optimal_matches


def mm_metric(pred_cc, gt_cc, metric='dsc'):
    """
    Calculate MaxMerge Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: Kofler, F., Möller, H., Buchner, J.A., de la Rosa, E., Ezhov, I., Rosier, M., Mekki, I., Shit, S., Negwer, M., Al-Maskari, R. and Ertürk, A., 2023. Panoptica--instance-wise evaluation of 3D semantic and instance segmentation maps. arXiv preprint arXiv:2312.02608.
    """
    # Extract the number of classes from the prediction tensor
    num_classes = pred_cc.shape[0]
    
    # Initialize counters for tracking overall performance
    total_metric_score = 0.0
    total_tps = 0  # True positives
    total_fps = 0  # False positives
    total_fns = 0  # False negatives
    
    # Process each class separately, skipping background (class 0)
    for cls in range(1, num_classes):
        # Handle special cases (empty predictions or ground truths)
        score, handled = _handle_empty_classes(pred_cc[cls], gt_cc[cls])
        if handled:
            # Update counters based on special case outcomes
            total_tps += 1 if score == 1.0 else 0  # Perfect match implies true positive
            total_fps += 1 if score == 0.0 and pred_cc[cls].sum() > 0 else 0  # Prediction without ground truth
            total_fns += 1 if score == 0.0 and gt_cc[cls].sum() > 0 else 0  # Ground truth without prediction
            continue

        # Initialize class-level counters
        cls_tp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_fp = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_fn = torch.zeros(0, dtype=torch.int64, device=pred_cc[cls].device)
        cls_metric_score = 0.0

        # Create a bipartite graph of matches between predictions and ground truths
        # This includes potential merged predictions (many-to-one matches)
        matches = create_match_dict(pred_cc[cls], gt_cc[cls], metric=METRIC_FUNCS[metric])
        match_data = get_all_matches(matches)

        # Count false positives (predictions without matching ground truth)
        cls_fp = sum(1 for pred, gt, _ in match_data if gt is None)
        # Count false negatives (ground truths without matching predictions)
        cls_fn = sum(1 for pred, gt, _ in match_data if pred is None)

        # Use Hungarian algorithm to find optimal one-to-one matching
        # that maximizes the overall matching score
        optimal_matches = optimal_matching(match_data)

        # Number of true positives is the number of optimal matches
        cls_tp = len(optimal_matches)
        
        # Early return if no true positives - indicates complete failure for this class
        if cls_tp == 0:
            return 0.0
        
        # Sum up the scores for all optimal matches
        cls_metric_score = sum(score for _, _, score in optimal_matches)

        # Update global counters
        total_tps += cls_tp
        total_fps += cls_fp
        total_fns += cls_fn
        total_metric_score += cls_metric_score

    # Normalize the score by total number of instances (TP + FP + FN)
    # This accounts for both precision and recall aspects
    mean_metric_score = total_metric_score / (total_fns + total_tps + total_fps)

    return mean_metric_score