import torch
from monai.metrics import compute_surface_dice

def multiclass_nsd(pred, gt, exclude_background=True):
    """
    Calculates the Normalized Surface Dice (NSD) between prediction and ground truth masks.
    
    The function first adds batch and channel dimensions to the input tensors as required by MONAI,
    then computes the surface Dice score with a tolerance of 1.0.
    
    Args:
        pred (torch.Tensor): Predicted binary segmentation mask
        gt (torch.Tensor): Ground truth binary segmentation mask
        exclude_background (bool): If True, excludes the background class (assumed to be class 0).
        
    Returns:
        float: The mean Normalized Surface Dice score. Returns 0 if the computation results in NaN.
    """

    #creating thresholds
    num_classes = gt.shape[0] - (1 if exclude_background else 0)  # Adjust for background exclusion
    threshold_val = 1  # Threshold value for binary segmentation
    thresholds = [threshold_val] * num_classes

    # Add batch and channel dimensions to both tensors as required by MONAI
    pred = pred.unsqueeze(0)
    gt = gt.unsqueeze(0)

    distance = compute_surface_dice(pred, gt, thresholds)

    return 0 if torch.isnan(distance).any() else distance.mean().item()
