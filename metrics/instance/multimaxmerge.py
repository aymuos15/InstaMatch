import torch

from metrics.tools.connected_components import gpu_connected_components
from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes, _calculate_final_score

def mmm_metric(pred, gt, metric='dsc'):
    """
    Calculate MultiMaxMerge Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One Hot Encoded Tensor | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One Hot Encoded Tensor | (C, H, W) or (C, H, W, D).

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: 
    """
    # Extract the number of classes from prediction tensor
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    # Process each class separately, skipping background (class 0)
    for cls in range(1, num_classes):
        # Handle special cases (empty predictions or ground truths)
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        # Create a binary union mask where either prediction or ground truth has foreground
        overlay = pred[cls] + gt[cls]
        overlay[overlay > 0] = 1
        # Identify spatially connected regions using connected components analysis
        labeled_array, num_features = gpu_connected_components(overlay)

        # Store metric scores for each cluster and build optimized predictions
        cls_metric_score = []
        optimized_class_pred = torch.zeros_like(pred[cls])

        # Process each connected component (cluster) separately
        for cls_instance in range(1, num_features + 1):
            # Extract the current cluster region
            cluster_mask = (labeled_array == cls_instance)
            # Isolate prediction and ground truth for this specific cluster
            pred_cluster = pred[cls] * cluster_mask
            gt_cluster = gt[cls] * cluster_mask

            # Find optimal subset of predicted components that maximizes the metric
            # This is a key part of the MultiMaxMerge approach - optimizing each cluster
            optimized_pred_cluster, _ = optimize_dice_for_cluster(pred_cluster, gt_cluster, metric=metric)
            # Accumulate optimized predictions for the entire class
            optimized_class_pred += optimized_pred_cluster
            # Calculate metric for this optimized cluster
            cls_metric_score.append(METRIC_FUNCS[metric](optimized_pred_cluster, gt_cluster))

        # Ensure binary prediction by clamping values between 0 and 1
        optimized_class_pred = torch.clamp(optimized_class_pred, 0, 1)
        # Calculate mean score across all clusters in this class
        total_metric_score += torch.mean(torch.tensor(cls_metric_score))

    # Normalize the score and convert to Python scalar
    return _calculate_final_score(total_metric_score, num_classes).item()

def optimize_dice_for_cluster(pred_cluster, gt_cluster, metric='dsc'):
    """
    Find optimal connected components in pred_cluster to maximize the metric score with gt_cluster.
    
    This function performs an exhaustive search over all possible combinations of connected
    components in the prediction to find the subset that maximizes the evaluation metric.
    It addresses problems like over-segmentation by testing whether merging or removing
    certain components improves the overall score.
    """
    # Handle edge cases: if either prediction or ground truth is empty
    if pred_cluster.sum() == 0 or gt_cluster.sum() == 0:
        # No optimization needed, return original with its score
        return pred_cluster, METRIC_FUNCS[metric](pred_cluster, gt_cluster)
    
    # Perform connected components analysis on the prediction to identify distinct regions
    labeled_pred, num_components = gpu_connected_components(pred_cluster)
    
    # If prediction has 0 or 1 component, no optimization is possible
    if num_components <= 1:
        return pred_cluster, METRIC_FUNCS[metric](pred_cluster, gt_cluster)
    
    # Get unique component labels, ignoring background (0)
    component_labels = torch.unique(labeled_pred)
    component_labels = component_labels[component_labels != 0]
    
    # Initialize tracking variables for best performance
    best_dice = 0.0
    best_pred = torch.zeros_like(pred_cluster)
    
    # First check the full prediction (using all components together)
    # This serves as our baseline to improve upon
    full_dice = METRIC_FUNCS[metric](pred_cluster, gt_cluster)
    if full_dice > best_dice:
        best_dice = full_dice
        best_pred = pred_cluster.clone()
    
    # Extract binary masks for each individual connected component
    component_masks = []
    for label in component_labels:
        # Create binary mask for this component
        mask = (labeled_pred == label).to(pred_cluster.dtype)
        component_masks.append(mask)
    
    # Exhaustive search: try all possible combinations of components
    # This is a key aspect of MultiMaxMerge - finding the optimal subset
    # of components that best matches the ground truth
    for k in range(1, len(component_masks) + 1):
        # Import combinations here to avoid circular imports
        from itertools import combinations
        # Generate all ways to select k components from the available components
        for combo in combinations(range(len(component_masks)), k):
            # Create a new prediction using only the selected components
            temp_pred = torch.zeros_like(pred_cluster)
            for idx in combo:
                temp_pred = temp_pred + component_masks[idx]
            
            # Ensure the result remains a binary mask
            temp_pred = torch.clamp(temp_pred, 0, 1)
            
            # Evaluate this combination using the specified metric
            dice = METRIC_FUNCS[metric](temp_pred, gt_cluster)
            
            # Keep track of the best performing combination
            if dice > best_dice:
                best_dice = dice
                best_pred = temp_pred.clone()
    
    # Return the optimal prediction and its score
    return best_pred, best_dice