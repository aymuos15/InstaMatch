import torch
import cupy as cp
import cucim.skimage.measure as cucim_measure

# Connected Components
def gpu_connected_components(img, connectivity=None):
    """
    PyTorch wrapper for calculating connected components on a GPU using cupy and cucim.skimage.
    #? From: https://github.com/aymuos15/GPU-Connected-Components
    
    Args:
        img (torch.Tensor): Input image.
    
    Input:
        connectivity (int, optional): Connectivity defining the neighborhood. Default is None.
    
    Output:
        torch.Tensor: Labeled image with each connected component having a unique label, of shape (H, W).
        int: Number of connected components found.
    """
    img_cupy = cp.asarray(img)
    labeled_img, num_features = cucim_measure.label(img_cupy, connectivity=connectivity, return_num=True)
    labeled_img_torch = torch.as_tensor(labeled_img, device=img.device)
    return labeled_img_torch, num_features