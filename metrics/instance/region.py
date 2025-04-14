import torch
from scipy.ndimage import distance_transform_edt

from metrics.tools.connected_components import gpu_connected_components
from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes, _calculate_final_score

def cc_metric(pred, gt, metric='dsc'):
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    for cls in range(1, num_classes):
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        cls_region_map, num_cls_features = get_gt_regions(gt[cls], pred.device)
        cls_metric_scores = []  # Collect scores for all regions in this class
        for cls_region_label in range(1, num_cls_features + 1):
            cls_mask = (cls_region_map == cls_region_label)
            pred_region = pred[cls][cls_mask]
            gt_region = gt[cls][cls_mask]

            # Clip values from 0 to 1
            pred_region = torch.clamp(pred_region, 0, 1)
            gt_region = torch.clamp(gt_region, 0, 1)

            # For NSD, ensure the inputs are appropriately formatted
            if metric == 'nsd':
                pred_full = torch.zeros_like(cls_mask, dtype=torch.float32)
                gt_full = torch.zeros_like(cls_mask, dtype=torch.float32)

                # Place the region values back in their original positions
                pred_full[cls_mask] = pred_region.float()
                gt_full[cls_mask] = gt_region.float()

                cls_metric_score = METRIC_FUNCS[metric](pred_full, gt_full)
            else:
                cls_metric_score = METRIC_FUNCS[metric](pred_region.float(), gt_region.float())

            cls_metric_scores.append(cls_metric_score)

        # Calculate the mean score for the class and add to total
        if cls_metric_scores:
            mean_cls_score = torch.mean(torch.tensor(cls_metric_scores, device=pred.device))
            total_metric_score += mean_cls_score

    return _calculate_final_score(total_metric_score, num_classes).item()

def get_gt_regions(gt, device):
    _, num_features = gpu_connected_components(gt)
    distance_map = torch.zeros_like(gt, dtype=torch.float32)
    region_map = torch.zeros_like(gt, dtype=torch.long)

    for region_label in range(1, num_features + 1):
        region_mask = (gt == region_label)
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
