import torch
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment

from .helper import gpu_connected_components
from .base_metrics import dice, iou, nsd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Metric mapping dictionary
METRIC_FUNCTIONS = {
    'dice': dice,
    'iou': iou,
    'nsd': nsd
}

class InstanceMetric:
    """Base class for instance-sensitive metrics."""
    
    def __init__(self, metric='dice'):
        """
        Initialize the instance metric.
        
        Args:
            metric (str or callable): The metric function to use for comparison
        """
        if isinstance(metric, str):
            self.metric_func = METRIC_FUNCTIONS.get(metric.lower(), dice)
        else:
            self.metric_func = metric
        
        self.metric_name = getattr(self.metric_func, '__name__', str(metric))
    
    def __call__(self, pred, gt):
        """Calculate the instance metric between prediction and ground truth."""
        raise NotImplementedError("Subclasses must implement __call__")
    
    def _get_binary_masks_for_class(self, pred, gt, class_id):
        """
        Extract binary masks for a specific class from prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs
            gt (torch.Tensor): Ground truth segmentation with instance IDs
            class_id (int): Class ID to extract
            
        Returns:
            tuple: (pred_mask, gt_mask) as binary float tensors
        """
        pred_mask = (pred == class_id).float()
        gt_mask = (gt == class_id).float()
        return pred_mask, gt_mask


class PanopticDice(InstanceMetric):
    """
    Implementation of Panoptic Dice metric.
    Inspired from: Kirillov et al., 2019. "Panoptic Segmentation"
    """
    
    def create_match_dict(self, pred_label_cc, gt_label_cc):
        """
        Creates a dictionary of matches between predicted and ground truth components.
        
        Args:
            pred_label_cc (torch.Tensor): Connected component labeled prediction.
            gt_label_cc (torch.Tensor): Connected component labeled ground truth.
        
        Returns:
            dict: Dictionary containing pred_to_gt, gt_to_pred mappings and metric scores.
        """
        pred_to_gt = {}
        gt_to_pred = {}
        metric_scores = {}

        pred_labels = torch.unique(pred_label_cc)[1:]  # Exclude background (0)
        gt_labels = torch.unique(gt_label_cc)[1:]  # Exclude background (0)

        pred_masks = {label.item(): pred_label_cc == label for label in pred_labels}
        gt_masks = {label.item(): gt_label_cc == label for label in gt_labels}

        for pred_item, pred_mask in pred_masks.items():
            for gt_item, gt_mask in gt_masks.items():
                if torch.any(torch.logical_and(pred_mask, gt_mask)):
                    pred_to_gt.setdefault(pred_item, []).append(gt_item)
                    gt_to_pred.setdefault(gt_item, []).append(pred_item)
                    metric_scores[(pred_item, gt_item)] = self.metric_func(pred_mask, gt_mask)

        for gt_item in gt_labels:
            gt_to_pred.setdefault(gt_item.item(), [])
        for pred_item in pred_labels:
            pred_to_gt.setdefault(pred_item.item(), [])

        return {"pred_to_gt": pred_to_gt, "gt_to_pred": gt_to_pred, "metric_scores": metric_scores}
    
    def get_all_matches(self, matches):
        """
        Extracts all matches from match dictionary into a list of tuples.
        
        Args:
            matches (dict): Dictionary containing match information.
        
        Returns:
            list: List of (pred, gt, metric_score) tuples for all matches.
        """
        match_data = []

        for gt, preds in matches["gt_to_pred"].items():
            if not preds:
                match_data.append((None, gt, 0.0))
            else:
                for pred in preds:
                    metric_score = matches["metric_scores"].get((pred, gt), 0.0)
                    match_data.append((pred, gt, metric_score))

        for pred, gts in matches["pred_to_gt"].items():
            if not gts:
                match_data.append((pred, None, 0.0))

        return match_data
    
    def optimal_matching(self, match_data):
        """
        Performs optimal assignment between prediction and ground truth components.
        
        Args:
            match_data (list): List of (pred, gt, score) tuples.
        
        Returns:
            list: Optimal matches as (pred, gt, score) tuples.
        """
        predictions = set()
        ground_truths = set()
        valid_matches = []

        for pred, gt, score in match_data:
            if pred is not None and gt is not None:
                predictions.add(pred)
                ground_truths.add(gt)
                valid_matches.append((pred, gt, score))

        pred_to_index = {pred: i for i, pred in enumerate(predictions)}
        gt_to_index = {gt: i for i, gt in enumerate(ground_truths)}

        cost_matrix = torch.ones((len(predictions), len(ground_truths)))

        for pred, gt, score in valid_matches:
            i, j = pred_to_index[pred], gt_to_index[gt]
            cost_matrix[i, j] = 1 - score

        #todo: Use a torch variant here?
        row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

        optimal_matches = []
        for i, j in zip(row_ind, col_ind):
            pred = list(predictions)[i]
            gt = list(ground_truths)[j]
            score = 1 - cost_matrix[i, j].item()
            optimal_matches.append((pred, gt, score))
        
        metric_scores = [score for _, _, score in optimal_matches]

        return optimal_matches, metric_scores
    
    def __call__(self, pred, gt):
        """
        Calculate Panoptic Dice score between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs.
            gt (torch.Tensor): Ground truth segmentation with instance IDs.
        
        Returns:
            float: Panoptic Dice score between 0 and 1.
        """
        num_classes = int(max(pred.max().item(), gt.max().item()) + 1)

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_metric_score = 0

        for class_id in range(1, num_classes):  # Skip background class 0
            # Create binary masks for current class
            pred_mask, gt_mask = self._get_binary_masks_for_class(pred, gt, class_id)

            # If both are missing, skip this class
            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                continue

            pred_label_cc, num_pred_features = gpu_connected_components(pred_mask)
            gt_label_cc, num_gt_features = gpu_connected_components(gt_mask)

            matches = self.create_match_dict(pred_label_cc, gt_label_cc)
            match_data = self.get_all_matches(matches)

            fp = sum(1 for pred, gt, _ in match_data if gt is None)
            fn = sum(1 for pred, gt, _ in match_data if pred is None)

            optimal_matches, metric_scores = self.optimal_matching(match_data)

            tp = len(optimal_matches)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_metric_score += sum(metric_scores)

        pq = total_metric_score / (total_tp + total_fp + total_fn)

        return pq



class CCDice(InstanceMetric):
    """
    Implementation of CC-Dice metric.
    Jaus et al., "Every Component Counts: Rethinking the Measure of Success for Medical Semantic Segmentation in Multi-Instance Segmentation Tasks"
    """
    
    def get_gt_regions(self, gt):
        """
        Computes connected components and distance transform for ground truth regions.
        
        Args:
            gt (torch.Tensor): Ground truth binary mask.
        
        Returns:
            tuple: (region_map, num_features) where region_map assigns each pixel to a region
                  and num_features is the number of regions found.
        """
        # Step 1: Connected Components (using CPU as cc3d requires numpy)
        labeled_array, num_features = gpu_connected_components(gt)

        # Step 2: Compute distance transform for each region
        distance_map = torch.zeros_like(gt, dtype=torch.float32)
        region_map = torch.zeros_like(gt, dtype=torch.long)

        for region_label in range(1, num_features + 1):
            # Create region mask
            region_mask = (labeled_array == region_label)

            # Convert to numpy for distance transform, then back to torch
            # (since PyTorch doesn't have a direct equivalent of distance_transform_edt)
            #! Need to integrate: https://github.com/moyiliyi/GPU-Accelerated-Boundary-Losses-for-Medical-Segmentation
            region_mask_np = region_mask.cpu().numpy()
            distance = torch.from_numpy(
                distance_transform_edt(~region_mask_np)
            ).to(device)

            if region_label == 1 or distance_map.max() == 0:
                distance_map = distance
                region_map = region_label * torch.ones_like(gt, dtype=torch.long)
            else:
                update_mask = distance < distance_map
                distance_map[update_mask] = distance[update_mask]
                region_map[update_mask] = region_label

        return region_map, num_features
    
    def __call__(self, pred, gt):
        """
        Calculate CC-Dice score between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs.
            gt (torch.Tensor): Ground truth segmentation with instance IDs.
        
        Returns:
            torch.Tensor: CC-Dice score between 0 and 1.
        """
        num_classes = int(max(pred.max().item(), gt.max().item()) + 1)

        all_class_metric_scores = {}

        for class_id in range(1, num_classes):  # Skip background class 0
            # Create binary masks for current class
            pred_mask, gt_mask = self._get_binary_masks_for_class(pred, gt, class_id)

            # If both are missing, the score is 1
            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(1.0, device=pred.device)
                continue

            # If either is missing, the score is 0
            if pred_mask.sum() == 0 or gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(0.0, device=pred.device)
                continue

            # Get ground truth regions
            region_map, num_features = self.get_gt_regions(gt_mask)

            # Initialize a tensor to store metric scores
            metric_scores = torch.zeros(num_features, device=pred.device)

            for region_label in range(1, num_features + 1):
                region_mask = (region_map == region_label)
                pred_region = pred_mask[region_mask]
                gt_region = gt_mask[region_mask]

                # For NSD, we need to ensure the inputs are appropriately formatted
                if self.metric_name == 'nsd':
                    # Create a tensor with the same shape as the region mask
                    pred_full = torch.zeros_like(region_mask, dtype=torch.float32)
                    gt_full = torch.zeros_like(region_mask, dtype=torch.float32)
                    
                    # Place the region values back in their original positions
                    pred_full[region_mask] = pred_region
                    gt_full[region_mask] = gt_region
                    
                    metric_score = self.metric_func(pred_full, gt_full)
                else:
                    metric_score = self.metric_func(pred_region, gt_region)
                    
                # Subtract 1 from region_label since tensor is 0-indexed
                metric_scores[region_label - 1] = metric_score

            all_class_metric_scores[class_id] = torch.mean(metric_scores)

        # Return mean of all class-wise scores
        return torch.mean(torch.stack(list(all_class_metric_scores.values())))


class ClusterDice(InstanceMetric):
    """
    Implementation of Cluster Dice metric.
    Kundu et al., 2024. "Cluster Dice: A simple approach for many-to-many instance matching scheme"
    """
    
    def __call__(self, pred, gt):
        """
        Calculate Cluster Dice score between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs.
            gt (torch.Tensor): Ground truth segmentation with instance IDs.
        
        Returns:
            torch.Tensor: Cluster Dice score between 0 and 1.
        """
        num_classes = int(max(pred.max().item(), gt.max().item()) + 1)

        all_class_metric_scores = {}

        for class_id in range(1, num_classes):  # Skip background class 0
            # Create binary masks for current class
            pred_mask, gt_mask = self._get_binary_masks_for_class(pred, gt, class_id)

            # If both are missing, the score is 1
            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(1.0, device=pred.device)
                continue

            # If either is missing, the score is 0
            if pred_mask.sum() == 0 or gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(0.0, device=pred.device)
                continue

            # Step 1: Create the overlay
            overlay = pred_mask + gt_mask
            overlay[overlay > 0] = 1

            # Step 2: Cluster the overlay
            labeled_array, num_features = gpu_connected_components(overlay)

            # Step 3: Calculate metric scores for each cluster
            class_metric_scores = []
            
            for i in range(1, num_features + 1):  # Start from 1 to exclude background
                # Create masks for current cluster
                cluster_mask = (labeled_array == i)
                pred_cluster = pred_mask * cluster_mask
                gt_cluster = gt_mask * cluster_mask
                
                # For NSD, we need to ensure the inputs are appropriately formatted
                if self.metric_name == 'nsd':
                    # We keep the full spatial context for NSD calculation
                    metric_score = self.metric_func(pred_cluster, gt_cluster)
                else:
                    metric_score = self.metric_func(pred_cluster, gt_cluster)
                    
                class_metric_scores.append(metric_score)
            
            all_class_metric_scores[class_id] = torch.mean(torch.stack(class_metric_scores))

        # Return mean of all class-wise metric scores
        return torch.mean(torch.stack(list(all_class_metric_scores.values())))


class LesionWiseDice(InstanceMetric):
    """
    Implementation of Lesion-wise Dice metric.
    Optimized version based on BraTS-Mets 2023/24 Challenge metrics.
    """
    
    def __call__(self, pred, gt):
        """
        Calculate Lesion-wise Dice score between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs.
            gt (torch.Tensor): Ground truth segmentation with instance IDs.
        
        Returns:
            torch.Tensor: Lesion-wise Dice score between 0 and 1.
        """
        num_classes = int(max(pred.max().item(), gt.max().item()) + 1)
        
        total_lesion_metric_scores = 0.0
        total_gt_lesions = 0
        total_fps = 0
        
        for class_id in range(1, num_classes):  # Skip background class 0
            # Create binary masks for current class
            pred_mask, gt_mask = self._get_binary_masks_for_class(pred, gt, class_id)
            
            # Skip if both are empty
            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                continue
                
            # Process connected components for this class
            pred_label_cc, _ = gpu_connected_components(pred_mask)
            gt_label_cc, _ = gpu_connected_components(gt_mask)

            num_class_gt_lesions = torch.unique(gt_label_cc[gt_label_cc != 0]).size(0)
            total_gt_lesions += num_class_gt_lesions

            class_lesion_metric_scores = 0
            class_tp = torch.tensor([]).to(device)

            for gtcomp in range(1, int(gt_label_cc.max().item()) + 1):
                gt_tmp = (gt_label_cc == gtcomp)
                if gt_tmp.sum() == 0:  # Skip if this component doesn't exist
                    continue
                    
                intersecting_cc = torch.unique(pred_label_cc[gt_tmp])
                intersecting_cc = intersecting_cc[intersecting_cc != 0]

                if len(intersecting_cc) > 0:
                    pred_tmp = torch.zeros_like(pred_label_cc, dtype=torch.bool)
                    pred_tmp[torch.isin(pred_label_cc, intersecting_cc)] = True
                    
                    # For NSD, we need to ensure inputs are correctly formatted
                    if self.metric_name == 'nsd':
                        pred_tmp = pred_tmp.float()  # Convert to float
                        gt_tmp = gt_tmp.float()  # Convert to float
                        
                    metric_score = self.metric_func(pred_tmp, gt_tmp)
                    class_lesion_metric_scores += metric_score
                    class_tp = torch.cat([class_tp, intersecting_cc])

            # Count false positives for this class
            class_mask = (pred_label_cc != 0) & (~torch.isin(pred_label_cc, class_tp))
            class_fp = torch.unique(pred_label_cc[class_mask], sorted=True)
            class_fp = class_fp[class_fp != 0]
            total_fps += len(class_fp)
            
            total_lesion_metric_scores += class_lesion_metric_scores
        
        # Calculate final score across all classes
        if total_gt_lesions == 0:
            return torch.tensor(0.0, device=device)
        else:
            return total_lesion_metric_scores / (total_gt_lesions + total_fps)

class MaximisedMergeDice(InstanceMetric):
    """
    Implementation of Optimized Cluster Dice metric.
    Extends Cluster Dice by Kundu et al., 2024 with component optimization.
    """
    
    def _optimize_dice_for_cluster(self, pred_cluster, gt_cluster):
        """Find optimal connected components in pred_cluster to maximize Dice score with gt_cluster"""
        # If either is empty, return the original prediction and its Dice score
        if pred_cluster.sum() == 0 or gt_cluster.sum() == 0:
            return pred_cluster, self.metric_func(pred_cluster, gt_cluster)
        
        # Find connected components in pred_cluster
        labeled_pred, num_components = gpu_connected_components(pred_cluster)
        
        if num_components <= 1:
            # If there's only one or zero components, just return the original
            return pred_cluster, self.metric_func(pred_cluster, gt_cluster)
        
        # Get component labels (excluding background/0)
        component_labels = torch.unique(labeled_pred)
        component_labels = component_labels[component_labels != 0]
        
        # Initialize with original values
        best_dice = self.metric_func(pred_cluster, gt_cluster)
        best_pred = pred_cluster.clone()
        
        # Test individual components
        for label in component_labels:
            # Create a mask with only this component
            temp_pred = torch.zeros_like(pred_cluster)
            temp_mask = (torch.tensor(labeled_pred == label, device=pred_cluster.device)).float()
            temp_pred += temp_mask
            
            # Calculate Dice score
            dice = self.metric_func(temp_pred, gt_cluster)
            
            # Update if better
            if dice > best_dice:
                best_dice = dice
                best_pred = temp_pred.clone()
        
        return best_pred, best_dice
    
    def __call__(self, pred, gt):
        """
        Calculate Optimized Cluster Dice score between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs.
            gt (torch.Tensor): Ground truth segmentation with instance IDs.
        
        Returns:
            torch.Tensor: Optimized Cluster Dice score between 0 and 1.
        """
        num_classes = int(max(pred.max().item(), gt.max().item()) + 1)
        all_class_metric_scores = {}
        
        # Store optimized predictions for visualization if needed
        optimized_preds = {}

        for class_id in range(1, num_classes):  # Skip background class 0
            # Create binary masks for current class
            pred_mask, gt_mask = self._get_binary_masks_for_class(pred, gt, class_id)

            # If both are missing, the score is 1
            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(1.0, device=pred.device)
                continue

            # If either is missing, the score is 0
            if pred_mask.sum() == 0 or gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(0.0, device=pred.device)
                continue

            # Step 1: Create the overlay
            overlay = pred_mask + gt_mask
            overlay[overlay > 0] = 1

            # Step 2: Cluster the overlay
            labeled_array, num_features = gpu_connected_components(overlay)

            # Step 3: Calculate metric scores for each cluster with optimization
            class_metric_scores = []
            optimized_class_pred = torch.zeros_like(pred_mask)
            
            for i in range(1, num_features + 1):  # Start from 1 to exclude background
                # Create masks for current cluster
                cluster_mask = (labeled_array == i)
                pred_cluster = pred_mask * cluster_mask
                gt_cluster = gt_mask * cluster_mask
                
                # Optimize the prediction components within this cluster
                optimized_pred_cluster, dice_score = self._optimize_dice_for_cluster(pred_cluster, gt_cluster)
                
                # Add the optimized prediction to our full optimized prediction
                optimized_class_pred += optimized_pred_cluster
                
                # Store the optimized Dice score
                class_metric_scores.append(dice_score)
            
            # Store the class-wise mean score and optimized prediction
            all_class_metric_scores[class_id] = torch.mean(torch.stack(class_metric_scores))
            optimized_preds[class_id] = optimized_class_pred

        # Return mean of all class-wise metric scores and optimized predictions
        mean_score = torch.mean(torch.stack(list(all_class_metric_scores.values())))
        
        # Create a full optimized prediction by combining class-wise optimized predictions
        optimized_full_pred = torch.zeros_like(pred)
        for class_id, opt_pred in optimized_preds.items():
            # Add the class ID to non-zero elements in the optimized prediction
            class_indices = (opt_pred > 0)
            optimized_full_pred[class_indices] = class_id
        
        return mean_score

class BlobDice(InstanceMetric):
    """
    Implementation of Blob Dice metric.
    
    For each connected component in the ground truth, creates a mask where only that component
    is treated as foreground (1) and all other ground truth components are treated as background (0).
    The prediction is masked with this, and a metric is calculated for each blob.
    The final score is the mean of all blob-wise scores.
    """
    
    def __call__(self, pred, gt):
        """
        Calculate Blob Dice score between prediction and ground truth.
        
        Args:
            pred (torch.Tensor): Predicted segmentation with instance IDs.
            gt (torch.Tensor): Ground truth segmentation with instance IDs.
        
        Returns:
            torch.Tensor: Blob Dice score between 0 and 1.
        """
        num_classes = int(max(pred.max().item(), gt.max().item()) + 1)
        
        all_class_metric_scores = {}
        
        for class_id in range(1, num_classes):  # Skip background class 0
            # Create binary masks for current class
            pred_mask, gt_mask = self._get_binary_masks_for_class(pred, gt, class_id)
            
            # If both are missing, the score is 1
            if pred_mask.sum() == 0 and gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(1.0, device=pred.device)
                continue
                
            # If either is missing, the score is 0
            if pred_mask.sum() == 0 or gt_mask.sum() == 0:
                all_class_metric_scores[class_id] = torch.tensor(0.0, device=pred.device)
                continue
            
            # Get connected components for ground truth
            gt_label_cc, _ = gpu_connected_components(gt_mask)
            
            # Get unique blob labels (excluding background/0)
            unique_labels = torch.unique(gt_label_cc)
            unique_labels = unique_labels[unique_labels != 0]
            
            blob_scores = []
            
            for blob_label in unique_labels:
                # Create mask where this blob is foreground and everything else is background
                label_mask = gt_label_cc > 0
                label_mask = ~label_mask
                label_mask[gt_label_cc == blob_label] = 1
                
                # Create binary mask for this blob
                blob_gt = (gt_label_cc == blob_label).float()
                
                # Apply mask to prediction
                masked_pred = pred_mask * label_mask
                
                # For NSD, we need to ensure the inputs are appropriately formatted
                if self.metric_name == 'nsd':
                    blob_score = self.metric_func(masked_pred, blob_gt)
                else:
                    blob_score = self.metric_func(masked_pred, blob_gt)
                
                blob_scores.append(blob_score)
            
            # Calculate mean score for this class
            if blob_scores:
                all_class_metric_scores[class_id] = torch.mean(torch.stack(blob_scores))
            else:
                all_class_metric_scores[class_id] = torch.tensor(0.0, device=pred.device)
        
        # Return mean of all class-wise scores
        return torch.mean(torch.stack(list(all_class_metric_scores.values())))

# For backward compatibility with function-based interface
def panoptic_dice(pred, gt, metric='dice'):
    """Function wrapper for PanopticDice class"""
    return PanopticDice(metric)(pred, gt)

def cc_dice(pred, gt, metric='dice'):
    """Function wrapper for CCDice class"""
    return CCDice(metric)(pred, gt)

def cluster_dice(pred, gt, metric='dice'):
    """Function wrapper for ClusterDice class"""
    return ClusterDice(metric)(pred, gt)

def lesion_wise_dice(pred, gt, metric='dice'):
    """Function wrapper for LesionWiseDice class"""
    return LesionWiseDice(metric)(pred, gt)

def maximised_merge_dice(pred, gt, metric='dice'):
    """Function wrapper for MaximisedMergeDice class"""
    return MaximisedMergeDice(metric)(pred, gt)

# Function wrapper for backward compatibility
def blob_dice(pred, gt, metric='dice'):
    """Function wrapper for BlobDice class"""
    return BlobDice(metric)(pred, gt)