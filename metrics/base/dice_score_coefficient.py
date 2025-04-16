import torch

def dsc(im1, im2):
    """
    Calculates the Dice Score Coefficient (DSC) between two binary images.
    
    Input:
        im1 (torch.Tensor): Binary image | (H, W) or (H, W, D).
        im2 (torch.Tensor): Binary image | (H, W) or (H, W, D).
    
    Returns:
        torch.Tensor: DSC between 0 and 1.
    """
    intersection = torch.sum(im1 * im2)
    sum_im1 = torch.sum(im1)
    sum_im2 = torch.sum(im2)
    return 2.0 * intersection / (sum_im1 + sum_im2)