import torch
from monai.metrics import compute_surface_dice

def multiclass_nsd(pred, gt, exclude_background=True):
    """
    Calculates the Normalized Surface Dice (NSD) for Multiple Classes.
    
    Input:
        pred (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
    
    Args:
        exclude_background (bool): If True, excludes the background class (assumed to be class 0).
    
    Output:
        float: The mean Normalized Surface Dice score across all classes (excluding background if specified).
        Returns 0 if the computation results in NaN.
        
    Note:
        Do not confuse this with the base metrics. Those are only for binary masks.
        This function uses MONAI's compute_surface_dice with a tolerance of 1.0.
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
