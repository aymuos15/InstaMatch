import torch

def dsc(im1, im2):
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