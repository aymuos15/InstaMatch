import torch

def multiclass_iou(pred, gt, exclude_background=True):
    """
    Calculates the Intersection over Union (IoU) for Multiple Classes.
    
    Input:
        pred (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
        gt (torch.Tensor): One-hot encoded Tensor with Channel wise Connected Cmponents | (C, H, W) or (C, H, W, D).
    
    Args:
        exclude_background (bool): If True, excludes the background class (assumed to be class 0).
    
    Output:
        torch.Tensor: Mean IoU score across all classes (excluding background if specified).
        
    Note:
        Do not confuse this with the base metrics. Those are only for binary masks.
    """
    
    if exclude_background:
        pred = pred[1:]  # Exclude background class (class 0)
        gt = gt[1:]      # Exclude background class (class 0)
    
    # Calculate intersection and union for each class (across height and width)
    intersection = torch.sum(pred * gt, dim=(1, 2))  # Sum over H, W for each class
    union = torch.sum(pred, dim=(1, 2)) + torch.sum(gt, dim=(1, 2)) - intersection  # Union = sum_pred + sum_gt - intersection
    
    # Calculate IoU for each class
    iou_scores = intersection / union
    
    # Return the mean IoU across all classes
    return iou_scores.mean()
