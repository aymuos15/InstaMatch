import torch
import torch.nn.functional as F

def dilate_mask(mask):
    """
    Dilate a binary mask by 1 pixel
    
    Inputs:
        mask: Binary mask tensor
    
    Returns:
        dilated: Dilated binary mask
    """
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), device=mask.device)
    dilated = F.conv2d(mask_f, kernel, padding=1) > 0
    return dilated.squeeze(0).squeeze(0)