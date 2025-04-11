import torch

def multiclass_iou(pred, gt):
    """
    Calculates the Intersection over Union (IoU) for multi-class one-hot encoded predictions and ground truths.
    
    Args:
        pred (torch.Tensor): One-hot encoded predictions of shape (C, H, W).
        gt (torch.Tensor): One-hot encoded ground truths of shape (C, H, W).
    
    Output:
        torch.Tensor: Mean IoU score across all classes.
    """
    
    # Calculate intersection and union for each class (across height and width)
    intersection = torch.sum(pred * gt, dim=(1, 2))  # Sum over H, W for each class
    union = torch.sum(pred, dim=(1, 2)) + torch.sum(gt, dim=(1, 2)) - intersection  # Union = sum_pred + sum_gt - intersection
    
    # Calculate IoU for each class
    iou_scores = intersection / union
    
    # Return the mean IoU across all classes
    return iou_scores.mean()
