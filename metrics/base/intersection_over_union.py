import torch

def iou(im1, im2):
    """
    Calculates the Intersection over Union (IoU) between two binary images.
    
    Input:
        im1 (torch.Tensor): Binary image | (H, W) or (H, W, D).
        im2 (torch.Tensor): Binary image | (H, W) or (H, W, D).
    
    Returns:
        torch.Tensor: IoU score between 0 and 1.
    """
    intersection = torch.sum(im1 * im2)
    union = torch.sum(im1) + torch.sum(im2) - intersection
    return intersection / (union)  