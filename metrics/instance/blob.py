import torch

from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes, _calculate_final_score

def blob_metric(pred, gt, metric='dsc'):
    """
    Calculate Blob Metrics between predicted and ground truth segmentations.
    The base metric can be easily swapped out by changing the metric parameter.

    Input:
        pred (torch.Tensor): One Hot Encoded Tensor | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).

    Args:
        metric (str): The metric to use for evaluation. Default is 'dsc'.

    Returns:
        float: The average metric score across all classes.

    Reference: Kofler F, Shit S, Ezhov I, Fidon L, Horvath I, Al-Maskari R, Li HB, Bhatia H, Loehr T, Piraud M, Erturk A. Blob loss: Instance imbalance aware loss functions for semantic segmentation. InInternational Conference on Information Processing in Medical Imaging 2023 Jun 8 (pp. 755-767). Cham: Springer Nature Switzerland.
    """
    # Extract the number of classes from the one-hot encoded predictions
    num_classes = pred.shape[0]
    total_metric_score = 0.0

    # Skip background class (index 0) and calculate metrics for all foreground classes
    for cls in range(1, num_classes):
        # Handle special cases where a class might be empty in either prediction or ground truth
        score, handled = _handle_empty_classes(pred[cls], gt[cls])
        if handled:
            total_metric_score += score
            continue

        # Find all unique blob labels within this class (exclude background value 0)
        unique_cls_labels = torch.unique(gt[cls][gt[cls] != 0])
        # Initialize tensor to store metric scores for each blob in this class
        cls_metric_score = torch.zeros(unique_cls_labels.shape[0], device=pred.device)

        # Calculate metric for each individual blob within this class
        for i, cls_blob_label in enumerate(unique_cls_labels):
            # Create a mask where all foreground regions of this class are marked
            cls_label_mask = gt[cls] > 0
            # Invert the mask to focus on background
            cls_label_mask = ~cls_label_mask
            # Mark only the current blob as foreground in the mask
            cls_label_mask[gt[cls] == cls_blob_label] = 1

            # Create a binary mask for the current blob
            the_cls_label = gt[cls] == cls_blob_label
            the_cls_label_int = the_cls_label.int()
            # Isolate the predicted region for the current blob location
            masked_output = pred[cls] * cls_label_mask

            # Calculate the specified metric (e.g., DSC) for this individual blob
            cls_metric_score[i] = METRIC_FUNCS[metric](masked_output, the_cls_label_int)

        # Average the metric scores across all blobs in this class
        total_metric_score += torch.mean(cls_metric_score)
    
    # Normalize the score and convert to Python scalar
    return _calculate_final_score(total_metric_score, num_classes).item()