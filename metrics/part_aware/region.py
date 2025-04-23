import torch
from scipy.ndimage import distance_transform_edt

from metrics.tools.connected_components import gpu_connected_components
from metrics.tools.mask_dilation import dilate_mask
from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes

def stuff_cls_scores(pred_one_hot_cc, gt_one_hot_cc, metric='dsc', stuff_cls_list=None):
    """
    Calculate quality score for stuff classes (semantic segments)
    
    Inputs:
        pred_one_hot_cc: One-hot encoded prediction tensor
        gt_one_hot_cc: One-hot encoded ground truth tensor
    
    Args:
        metric: Metric name to use for comparison
        stuff_cls_list: List of stuff class indices to process
        
    Returns:
        mean_score: Mean quality score across all specified stuff classes
        
    Note: 
        If stuff_cls_list is None or empty, returns 0
    """
    total_score = 0.0
    
    if not stuff_cls_list:
        return 0.0
    
    for cls_idx in stuff_cls_list:
        pred_one_hot_cc[cls_idx] = torch.clamp(pred_one_hot_cc[cls_idx], 0, 1)
        gt_one_hot_cc[cls_idx] = torch.clamp(gt_one_hot_cc[cls_idx], 0, 1)
        score = METRIC_FUNCS[metric](pred_one_hot_cc[cls_idx], gt_one_hot_cc[cls_idx])
        total_score += score
    
    mean_score = total_score / len(stuff_cls_list)
    return mean_score

def process_part_metrics(region_mask, dilated_region_mask, pred_one_hot_cc, gt_one_hot_cc, part_cls, metric):
    """
    Process metrics for a specific part class within a region
    
    Inputs:
        region_mask: Binary mask for the current region
        dilated_region_mask: Dilated binary mask for better part detection
    
    Args:
        pred_one_hot_cc: One-hot encoded prediction tensor
        gt_one_hot_cc: One-hot encoded ground truth tensor
        part_cls: Part class index to process
        metric: Metric name to use for comparison
        
    Returns:
        part_score: Score for this part class if parts exist, None otherwise
        pred_part_mask: Binary mask for prediction parts
        gt_part_mask: Binary mask for ground truth parts
    """
    pred_part = pred_one_hot_cc[part_cls]
    gt_part = gt_one_hot_cc[part_cls]
    
    pred_part_mask = torch.zeros_like(region_mask, dtype=torch.bool)
    gt_part_mask = torch.zeros_like(region_mask, dtype=torch.bool)
    
    for part_label in torch.unique(pred_part)[1:]:
        part_mask = pred_part == part_label
        if torch.any(torch.logical_and(dilated_region_mask, part_mask)):
            pred_part_mask = torch.logical_or(pred_part_mask, part_mask)
    
    for part_label in torch.unique(gt_part)[1:]:
        part_mask = gt_part == part_label
        if torch.any(torch.logical_and(dilated_region_mask, part_mask)):
            gt_part_mask = torch.logical_or(gt_part_mask, part_mask)
    
    if torch.any(pred_part_mask) and torch.any(gt_part_mask):
        if metric == 'nsd':
            pred_full = torch.zeros_like(region_mask, dtype=torch.float32)
            gt_full = torch.zeros_like(region_mask, dtype=torch.float32)
            
            pred_full[pred_part_mask] = 1.0
            gt_full[gt_part_mask] = 1.0
            
            part_score = METRIC_FUNCS[metric](pred_full, gt_full)
        else:
            part_score = METRIC_FUNCS[metric](pred_part_mask.float(), gt_part_mask.float())
        
        return part_score, pred_part_mask, gt_part_mask
    
    return None, pred_part_mask, gt_part_mask

def process_region(region_label, region_mask, gt_cc, pred_one_hot_cc, gt_one_hot_cc, class_idx, metric, part_cls_list, results):
    """
    Process a single region for the region-based metric
    
    Inputs:
        region_label: Label of the current region
        region_mask: Binary mask for the current region
    
    Args:
        gt_cc: Ground truth connected components
        pred_one_hot_cc: One-hot encoded prediction tensor
        gt_one_hot_cc: One-hot encoded ground truth tensor
        class_idx: Class index being processed
        metric: Metric name to use for comparison
        part_cls_list: List of part class indices
        results: Dictionary to store results
        
    Returns:
        region_score: Score for this region
        region_part_scores: Dictionary of part scores for this region
    """
    pred_region = pred_one_hot_cc[class_idx][region_mask]
    gt_region = gt_one_hot_cc[class_idx][region_mask]

    pred_region = torch.clamp(pred_region, 0, 1)
    gt_region = torch.clamp(gt_region, 0, 1)
    
    if metric == 'nsd':
        pred_full = torch.zeros_like(region_mask, dtype=torch.float32)
        gt_full = torch.zeros_like(region_mask, dtype=torch.float32)
        
        pred_full[region_mask] = pred_region.float()
        gt_full[region_mask] = gt_region.float()
        
        region_score = METRIC_FUNCS[metric](pred_full, gt_full)
    else:
        region_score = METRIC_FUNCS[metric](pred_region.float(), gt_region.float())
    
    results["class_metrics"][class_idx]["region_scores"][region_label] = region_score.item()
    
    region_part_scores = {}
    if part_cls_list is not None:
        dilated_region_mask = dilate_mask(region_mask)
        
        for part_cls in part_cls_list:
            part_score, _, _ = process_part_metrics(
                region_mask, dilated_region_mask, 
                pred_one_hot_cc, gt_one_hot_cc, 
                part_cls, metric
            )
            
            if part_score is not None:
                if part_cls not in results["class_metrics"][class_idx]["part_scores"]:
                    results["class_metrics"][class_idx]["part_scores"][part_cls] = {}
                
                part_score_value = part_score.item()
                results["class_metrics"][class_idx]["part_scores"][part_cls][region_label] = part_score_value
                region_part_scores[part_cls] = part_score_value
    
    return region_score, region_part_scores

def PartCC(pred_one_hot_cc, gt_one_hot_cc, metric='dsc', thing_cls_list=None, part_cls_list=None):
    """
    Calculate region-based metrics between predicted and ground truth segmentations
    
    Inputs:
        pred_one_hot_cc: One-hot encoded prediction tensor (C, H, W) or (C, H, W, D)
        gt_one_hot_cc: One-hot encoded ground truth tensor (C, H, W) or (C, H, W, D)
    
    Args:
        metric: Metric name to use for comparison
        thing_cls_list: List of "thing" class indices to process
        part_cls_list: List of part class indices to evaluate within thing regions
        
    Returns:
        final_score: Combined thing and stuff score
        
    Note:
        Each ground truth region is evaluated separately against predictions
        All part instances within a single "thing" instance are treated collectively
    """
    results = {
        "class_metrics": {},
        "overall_score": 0.0
    }
    
    num_classes = pred_one_hot_cc.shape[0]
    if thing_cls_list is None:
        all_classes = set(range(1, num_classes))
        part_classes = set(part_cls_list or [])
        thing_cls_list = list(all_classes - part_classes)
    
    total_metric_score = []
    
    for class_idx in thing_cls_list:
        results["class_metrics"][class_idx] = {
            "region_scores": {},
            "part_scores": {}
        }
        
        score, handled = _handle_empty_classes(pred_one_hot_cc[class_idx], gt_one_hot_cc[class_idx])
        if handled:
            total_metric_score += score
            results["class_metrics"][class_idx]["score"] = score
            continue
        
        gt_cc, num_gt_regions = gpu_connected_components(gt_one_hot_cc[class_idx])
        
        for region_label in range(1, num_gt_regions + 1):
            region_mask = (gt_cc == region_label)
            
            region_score, region_part_scores = process_region(
                region_label, region_mask, gt_cc, 
                pred_one_hot_cc, gt_one_hot_cc, 
                class_idx, metric, part_cls_list, results
            )
            
            if region_part_scores:
                all_scores = [region_score.item()] + list(region_part_scores.values())
                combined_score = sum(all_scores) / len(all_scores)
                
                if "combined_scores" not in results["class_metrics"][class_idx]:
                    results["class_metrics"][class_idx]["combined_scores"] = {}
                results["class_metrics"][class_idx]["combined_scores"][region_label] = combined_score

                total_metric_score.append(combined_score)

            else:
                total_metric_score.append(region_score.item())
                
    stuff_score = stuff_cls_scores(pred_one_hot_cc, gt_one_hot_cc, metric, thing_cls_list)

    thing_score = sum(total_metric_score) / len(total_metric_score) if total_metric_score else 0.0

    return (stuff_score + thing_score) / 2.0 #! Simple 1:1 weighted average for now. This can be changed.

def get_gt_regions(gt, device):
    """
    Divide ground truth segmentation space into regions based on proximity to instances
    
    Inputs:
        gt: Ground truth segmentation for a single class
        device: Device to place tensors on
    
    Returns:
        region_map: Tensor where each pixel is labeled with the nearest region ID
        num_features: Number of distinct regions/connected components
        
    Note:
        Uses distance transforms to create a Voronoi-like partition of the image space
    """
    gt_cc, num_features = gpu_connected_components(gt)
    
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    region_map = torch.zeros_like(gt, dtype=torch.long)

    for region_label in range(1, num_features + 1):
        region_mask = (gt_cc == region_label)
        region_mask_np = region_mask.cpu().numpy()
        distance = torch.from_numpy(distance_transform_edt(~region_mask_np)).to(device)
        
        if region_label == 1 or distance_map.max() == 0:
            distance_map = distance
            region_map = region_label * torch.ones_like(gt, dtype=torch.long)
        else:
            update_mask = distance < distance_map
            distance_map[update_mask] = distance[update_mask]
            region_map[update_mask] = region_label

    return region_map, num_features