from monai.metrics import compute_surface_dice

def nsd(im1, im2):
    """
    Calculates the Normalized Surface Dice (NSD) between two binary images.
    
    Input:
        im1 (torch.Tensor): Binary image | (H, W) or (H, W, D).
        im2 (torch.Tensor): Binary image | (H, W) or (H, W, D).
    
    Returns:
        torch.Tensor: NSD score between 0 and 1.

    Note:
        This is a wrapper around MONAI's compute_surface_dice function.
    """

    # Add batch and channel dimensions to both tensors as required by MONAI
    pred = im1.unsqueeze(0).unsqueeze(0)
    gt = im2.unsqueeze(0).unsqueeze(0)

    distance = compute_surface_dice(pred, gt, [1.0])

    # if distance is Nan, return 0
    if distance != distance:
        return 0
    else:
        return distance