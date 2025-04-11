import torch

from metrics.tools.connected_components import gpu_connected_components
from metrics.instance.helper import METRIC_FUNCS
from metrics.instance.helper import _handle_empty_classes, _calculate_final_score

def mmm_metric(pred, gt, metric='dsc'):
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    for cls in range(1, num_classes):
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        overlay = pred[cls] + gt[cls]
        overlay[overlay > 0] = 1
        labeled_array, num_features = gpu_connected_components(overlay)

        cls_metric_score = []
        optimized_class_pred = torch.zeros_like(pred[cls])

        for cls_instance in range(1, num_features + 1):
            cluster_mask = (labeled_array == cls_instance)
            pred_cluster = pred[cls] * cluster_mask
            gt_cluster = gt[cls] * cluster_mask

            optimized_pred_cluster, _ = optimize_dice_for_cluster(pred_cluster, gt_cluster, metric=metric)
            optimized_class_pred += optimized_pred_cluster
            cls_metric_score.append(METRIC_FUNCS[metric](optimized_pred_cluster, gt_cluster))

        optimized_class_pred = torch.clamp(optimized_class_pred, 0, 1)
        total_metric_score += torch.mean(torch.tensor(cls_metric_score))

    return _calculate_final_score(total_metric_score, num_classes).item()

def optimize_dice_for_cluster(pred_cluster, gt_cluster, metric='dsc'):
    """Find optimal connected components in pred_cluster to maximize Dice score with gt_cluster"""
    # If either is empty, return the original prediction and its Dice score
    if pred_cluster.sum() == 0 or gt_cluster.sum() == 0:
        return pred_cluster, METRIC_FUNCS[metric](pred_cluster, gt_cluster)
    
    # Find connected components in pred_cluster
    labeled_pred, num_components = gpu_connected_components(pred_cluster)
    
    if num_components <= 1:
        # If there's only one or zero components, just return the original
        return pred_cluster, METRIC_FUNCS[metric](pred_cluster, gt_cluster)
    
    # Get component labels (excluding background/0)
    component_labels = torch.unique(labeled_pred)
    component_labels = component_labels[component_labels != 0]
    
    # Initialize with empty prediction
    best_dice = 0.0
    best_pred = torch.zeros_like(pred_cluster)
    
    # Get dice score for original full prediction
    full_dice = METRIC_FUNCS[metric](pred_cluster, gt_cluster)
    if full_dice > best_dice:
        best_dice = full_dice
        best_pred = pred_cluster.clone()
    
    # Create individual component masks
    component_masks = []
    for label in component_labels:
        mask = (labeled_pred == label).to(pred_cluster.dtype)
        component_masks.append(mask)
    
    # Try all possible combinations of components
    # Start with 1 component and go up to num_components
    for k in range(1, len(component_masks) + 1):
        # Get all combinations of k components
        from itertools import combinations
        for combo in combinations(range(len(component_masks)), k):
            # Create combined mask for this combination
            temp_pred = torch.zeros_like(pred_cluster)
            for idx in combo:
                temp_pred = temp_pred + component_masks[idx]
            
            # Ensure binary mask
            temp_pred = torch.clamp(temp_pred, 0, 1)
            
            # Calculate Dice score
            dice = METRIC_FUNCS[metric](temp_pred, gt_cluster)
            
            # Update if better
            if dice > best_dice:
                best_dice = dice
                best_pred = temp_pred.clone()
    
    return best_pred, best_dice