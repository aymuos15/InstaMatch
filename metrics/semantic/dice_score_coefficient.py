import torch

def multiclass_dsc(pred, gt, exclude_background=True):
    """
    Calculates the Dice Score Coefficient for Multiple Classes.
    
    Input:
        pred (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
    
    Args:
        exclude_background (bool): If True, excludes the background class (assumed to be class 0).
    
    Output:
        torch.Tensor: Mean Dice coefficient score across all classes (excluding background if specified).

    Note:
        Do not confuse this with the base metrics. Those are only for binary masks.
    """

    if exclude_background:
        pred = pred[1:]  # Exclude background class (class 0)
        gt = gt[1:]      # Exclude background class (class 0)

    # Calculate intersection and sum for each class (across height and width)
    intersection = torch.sum(pred * gt, dim=(1, 2))  # Sum over H, W for each class
    sum_pred = torch.sum(pred, dim=(1, 2))  # Sum over H, W for the predicted values
    sum_gt = torch.sum(gt, dim=(1, 2))  # Sum over H, W for the ground truth values
    
    # Calculate Dice coefficient for each class
    dice_scores = 2.0 * intersection / (sum_pred + sum_gt)
    
    # Return the mean Dice coefficient across all classes
    return dice_scores.mean()
