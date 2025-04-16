import torch

from metrics.tools.connected_components import gpu_connected_components
from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes, _calculate_final_score

def cluster_metric(pred, gt, metric='dsc'):
    """
    Calculate Cluster Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One Hot Encoded Tensor | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One Hot Encoded Tensor | (C, H, W) or (C, H, W, D).

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: Kundu, S.S., Kujawa, A., Ivory, M., Barfoot, T., Shapey, J. and Vercauteren, T., 2025, April. Cluster dice: a simple and fast approach for instance-based semantic segmentation evaluation via many-to-many matching. In Medical Imaging 2025: Computer-Aided Diagnosis (Vol. 13407, pp. 226-232). SPIE.
    """
    # Extract the number of classes from the one-hot encoded predictions
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    # Skip background class (index 0) and process each foreground class
    for cls in range(1, num_classes):  # exclude background
        # Check if either prediction or ground truth is empty for this class
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        # Create a binary union mask where either prediction or ground truth has foreground
        overlay = pred[cls] + gt[cls]
        overlay[overlay > 0] = 1
        # Run connected components on the union mask to identify distinct clusters
        # where predicted and ground truth regions interact
        labeled_array, num_features = gpu_connected_components(overlay)

        # Initialize tensor to store metric scores for each cluster
        cls_metric_score = torch.zeros(num_features, device=pred.device)
        # Calculate metrics for each independent cluster
        for cls_instance in range(1, num_features + 1):
            # Extract the current cluster region
            cluster_mask = (labeled_array == cls_instance)
            # Isolate the prediction and ground truth for this specific cluster
            pred_cluster = pred[cls] * cluster_mask
            gt_cluster = gt[cls] * cluster_mask
            # Calculate the specified metric (e.g., DSC) for this cluster
            cls_metric_score[cls_instance - 1] = METRIC_FUNCS[metric](pred_cluster, gt_cluster)

        # Average metric scores across all clusters in this class
        total_metric_score += torch.mean(cls_metric_score)

    # Normalize the score and convert to Python scalar
    return _calculate_final_score(total_metric_score, num_classes).item()
