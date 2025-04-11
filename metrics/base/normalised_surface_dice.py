from monai.metrics import compute_surface_dice

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