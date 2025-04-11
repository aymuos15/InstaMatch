import torch

def one_hot_encode(input_tensor, num_classes):
    """
    Perform one-hot encoding on a 2D or 3D tensor.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (H, W) or (H, W, D).
        num_classes (int): Number of classes for one-hot encoding.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (num_classes, H, W) or (num_classes, H, W, D).
    
    Note:
        this returns the background as the first channel.
    """
    if input_tensor.dim() == 2:
        # One-hot encode
        one_hot = torch.nn.functional.one_hot(input_tensor.long(), num_classes=num_classes)
        # Reshape to (num_classes, H, W)
        one_hot = one_hot.permute(2, 0, 1)
        
    elif input_tensor.dim() == 3:
        # One-hot encode
        one_hot = torch.nn.functional.one_hot(input_tensor.long(), num_classes=num_classes)
        # Reshape to (num_classes, H, W, D)
        one_hot = one_hot.permute(3, 0, 1, 2)
        
    else:
        raise ValueError("Input tensor must be 2D or 3D.")

    return one_hot