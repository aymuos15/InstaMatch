import torch 
from base_metrics import dice, iou, nsd

def dice_multiclass(pred, gt):
    """
    Calculates the Dice coefficient for each class in a multiclass segmentation task.
    
    Args:
        pred (torch.Tensor): Predicted segmentation map.
        gt (torch.Tensor): Ground truth segmentation map.
        
    Returns:
        torch.Tensor: Tensor of Dice scores for each class, or the mean score if reduce=True.
    """
    # Initialize list to store Dice scores
    dice_scores = []
    
    # Iterate over each class
    for c in range(1, int(pred.max().item()) + 1):
        # Create binary masks for the current class
        pred_c = (pred == c).float()
        gt_c = (gt == c).float()
        
        # Calculate Dice score for the current class
        dice_score = dice(pred_c, gt_c)
        dice_scores.append(dice_score)
    
    # Convert list to tensor
    scores = torch.tensor(dice_scores)
    return scores.mean() if len(scores) > 0 else torch.tensor(0.0)

def iou_multiclass(pred, gt):
    """
    Calculates the Intersection over Union (IoU) for each class in a multiclass segmentation task.
    
    Args:
        pred (torch.Tensor): Predicted segmentation map.
        gt (torch.Tensor): Ground truth segmentation map.
        
    Returns:
        torch.Tensor: Tensor of IoU scores for each class, or the mean score if reduce=True.
    """
    
    # Initialize list to store IoU scores
    iou_scores = []
    
    # Iterate over each class
    for c in range(1, int(pred.max().item()) + 1):
        # Create binary masks for the current class
        pred_c = (pred == c).float()
        gt_c = (gt == c).float()
        
        # Calculate IoU score for the current class
        iou_score = iou(pred_c, gt_c)
        iou_scores.append(iou_score)
    # Convert list to tensor
    scores = torch.tensor(iou_scores)
    return scores.mean() if len(scores) > 0 else torch.tensor(0.0)

def nsd_multiclass(pred, gt):
    """
    Calculates the Normalized Surface Dice (NSD) for each class in a multiclass segmentation task.
    
    Args:
        pred (torch.Tensor): Predicted segmentation map.
        gt (torch.Tensor): Ground truth segmentation map.
        
    Returns:
        torch.Tensor: Tensor of NSD scores for each class.
    """
    # Initialize list to store NSD scores
    nsd_scores = []
    
    # Iterate over each class
    for c in range(1, int(pred.max().item()) + 1):
        # Create binary masks for the current class
        pred_c = (pred == c).float()
        gt_c = (gt == c).float()
        
        # Calculate NSD score for the current class
        nsd_score = nsd(pred_c, gt_c)
        nsd_scores.append(nsd_score)

    # Convert list to tensor
    scores = torch.tensor(nsd_scores)

    return scores.mean() if len(scores) > 0 else torch.tensor(0.0)