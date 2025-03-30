import torch
from monai.metrics import compute_surface_dice

def dice(im1, im2):
    """
    Calculates the Dice coefficient between two binary images.
    
    Args:
        im1 (torch.Tensor): First binary image.
        im2 (torch.Tensor): Second binary image.
    
    Input:
        im1 (torch.Tensor): Binary image of shape (H, W) or (N, H, W).
        im2 (torch.Tensor): Binary image of shape (H, W) or (N, H, W).
    
    Output:
        torch.Tensor: Dice coefficient score between 0 and 1.
    """
    intersection = torch.sum(im1 * im2)
    sum_im1 = torch.sum(im1)
    sum_im2 = torch.sum(im2)
    return 2.0 * intersection / (sum_im1 + sum_im2)

def iou(im1, im2):
    """
    Calculates the Intersection over Union (IoU) between two binary images.
    
    Args:
        im1 (torch.Tensor): First binary image.
        im2 (torch.Tensor): Second binary image.
    
    Input:
        im1 (torch.Tensor): Binary image of shape (H, W) or (N, H, W).
        im2 (torch.Tensor): Binary image of shape (H, W) or (N, H, W).
    
    Output:
        torch.Tensor: IoU score between 0 and 1.
    """
    intersection = torch.sum(im1 * im2)
    union = torch.sum(im1) + torch.sum(im2) - intersection
    return intersection / union

def nsd(pred, gt):
    """
    Calculates the Normalized Surface Dice (NSD) between prediction and ground truth masks.
    
    The function first adds batch and channel dimensions to the input tensors as required by MONAI,
    then computes the surface Dice score with a tolerance of 1.0.
    
    Args:
        pred (torch.Tensor): Predicted binary segmentation mask
        gt (torch.Tensor): Ground truth binary segmentation mask
        
    Returns:
        float: The Normalized Surface Dice score. Returns 0 if the computation results in NaN.
    """

    # Add batch and channel dimensions to both tensors as required by MONAI
    pred = pred.unsqueeze(0).unsqueeze(0)
    gt = gt.unsqueeze(0).unsqueeze(0)

    distance = compute_surface_dice(pred, gt, [1.0])

    # if distance is Nan, return 0
    if distance != distance:
        return 0
    else:
        return distance
