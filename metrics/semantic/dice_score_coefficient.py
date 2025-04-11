import torch

def multiclass_dsc(pred, gt):
    """
    Calculates the Dice coefficient for multi-class one-hot encoded predictions and ground truths.
    
    Args:
        pred (torch.Tensor): One-hot encoded predictions of shape (C, H, W).
        gt (torch.Tensor): One-hot encoded ground truths of shape (C, H, W).
    
    Output:
        torch.Tensor: Mean Dice coefficient score across all classes.
    """

    # Calculate intersection and sum for each class (across height and width)
    intersection = torch.sum(pred * gt, dim=(1, 2))  # Sum over H, W for each class
    sum_pred = torch.sum(pred, dim=(1, 2))  # Sum over H, W for the predicted values
    sum_gt = torch.sum(gt, dim=(1, 2))  # Sum over H, W for the ground truth values
    
    # Calculate Dice coefficient for each class
    dice_scores = 2.0 * intersection / (sum_pred + sum_gt)
    
    # Return the mean Dice coefficient across all classes
    return dice_scores.mean()
