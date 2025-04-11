import torch
from .one_hot_encoding import one_hot_encode
from .connected_components import gpu_connected_components

def label(input_tensor, num_classes, connectivity=None):
    """
    Perform connected components analysis on a one-hot encoded tensor.

    Inputs:
        input_tensor (torch.Tensor): One-hot encoded tensor of shape (num_classes, H, W) or (num_classes, H, W, D).
        num_classes (int): Number of classes.

    Args:
        connectivity (int, optional): Connectivity defining the neighborhood. Default is None.

    Returns:
        torch.Tensor: Labeled tensor of shape (num_classes, H, W) or (num_classes, H, W, D).
        int: Number of connected components found.
    
    Note:
        this returns the background as the first channel.
    """

    ohe_tensor = one_hot_encode(input_tensor, num_classes)
    labeled_tensor = torch.zeros_like(ohe_tensor)

    for i in range(num_classes):
        # Perform connected components analysis on each class
        labeled_img, _ = gpu_connected_components(ohe_tensor[i], connectivity)
        labeled_tensor[i] = labeled_img
    
    return labeled_tensor