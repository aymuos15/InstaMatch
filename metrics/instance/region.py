import torch
from scipy.ndimage import distance_transform_edt

from metrics.tools.connected_components import gpu_connected_components
from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes, _calculate_final_score

def cc_metric(pred, gt, metric='dsc'):
    """
    Calculate Connected Components Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One Hot Encoded Tensor | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: Jaus, A., Seibold, C.M., Reiß, S., Marinov, Z., Li, K., Ye, Z., Krieg, S., Kleesiek, J. and Stiefelhagen, R., 2025, April. Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 39, No. 4, pp. 3904-3912).
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

        # Divide the ground truth into spatial regions using distance transforms
        # This partitions the space into regions based on proximity to different instances
        cls_region_map, num_cls_features = get_gt_regions(gt[cls], pred.device)
        
        # Store scores for all regions in this class
        cls_metric_scores = []
        
        # Evaluate each spatial region separately
        for cls_region_label in range(1, num_cls_features + 1):
            # Create a mask for the current region
            cls_mask = (cls_region_map == cls_region_label)
            
            # Extract prediction and ground truth values for this region only
            pred_region = pred[cls][cls_mask]
            gt_region = gt[cls][cls_mask]

            # Ensure binary values by clamping between 0 and 1
            pred_region = torch.clamp(pred_region, 0, 1)
            gt_region = torch.clamp(gt_region, 0, 1)

            # Special handling for Normalized Surface Dice metric
            # NSD requires full spatial tensors with distance transforms
            if metric == 'nsd':
                # Create empty tensors matching the region shape
                pred_full = torch.zeros_like(cls_mask, dtype=torch.float32)
                gt_full = torch.zeros_like(cls_mask, dtype=torch.float32)

                # Project the region values back to their original spatial positions
                pred_full[cls_mask] = pred_region.float()
                gt_full[cls_mask] = gt_region.float()

                # Calculate NSD on full spatial tensors
                cls_metric_score = METRIC_FUNCS[metric](pred_full, gt_full)
            else:
                # For other metrics, calculate directly on the region values
                cls_metric_score = METRIC_FUNCS[metric](pred_region.float(), gt_region.float())

            # Store the score for this region
            cls_metric_scores.append(cls_metric_score)

        # Average the scores across all regions in this class
        if cls_metric_scores:
            mean_cls_score = torch.mean(torch.tensor(cls_metric_scores, device=pred.device))
            total_metric_score += mean_cls_score

    # Normalize the score and convert to Python scalar
    return _calculate_final_score(total_metric_score, num_classes).item()

def get_gt_regions(gt, device):
    """
    Divides the ground truth segmentation space into regions based on proximity to instances.
    
    This function uses distance transforms to create a Voronoi-like partition of the image space,
    where each pixel is assigned to the closest ground truth instance.
    
    Args:
        gt (torch.Tensor): Ground truth segmentation for a single class
        device (torch.device): Device to place tensors on
    
    Returns:
        tuple: (region_map, num_features)
            - region_map: Tensor where each pixel is labeled with the nearest region ID
            - num_features: Number of distinct regions/connected components
    """
    # Identify connected components in the ground truth
    _, num_features = gpu_connected_components(gt)
    
    # Initialize distance map (stores distance to nearest instance)
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    # Initialize region map (stores region ID for each pixel)
    region_map = torch.zeros_like(gt, dtype=torch.long)

    # Process each connected component (region) in the ground truth
    for region_label in range(1, num_features + 1):
        # Create binary mask for current region
        region_mask = (gt == region_label)
        # Convert to NumPy for scipy distance transform calculation
        region_mask_np = region_mask.cpu().numpy()
        # Calculate Euclidean distance transform from each pixel to the nearest region boundary
        # Note: We use ~region_mask_np to get distances from outside the region
        distance = torch.from_numpy(distance_transform_edt(~region_mask_np)).to(device)
        
        # For the first region or if we haven't set distances yet, initialize directly
        if region_label == 1 or distance_map.max() == 0:
            distance_map = distance
            region_map = region_label * torch.ones_like(gt, dtype=torch.long)
        else:
            # For subsequent regions, update pixels that are closer to this region
            # than to any previously processed region
            update_mask = distance < distance_map
            distance_map[update_mask] = distance[update_mask]
            region_map[update_mask] = region_label

    return region_map, num_features
