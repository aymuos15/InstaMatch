import torch
from scipy.optimize import linear_sum_assignment

from metrics.tools.mask_dilation import dilate_mask
from metrics.tools.utils import METRIC_FUNCS
from metrics.tools.utils import _handle_empty_classes

def create_match_dict(pred_one_hot_cc, gt_one_hot_cc, metric=None, thing_cls_list=None, part_cls_list=None):
    """
    Create matching dictionaries between predicted and ground truth connected components,
    organized by class. Also tracks part components that belong to each panoptic component 
    and evaluates them collectively.
    
    Args:
        pred_one_hot_cc: One-hot encoded tensor with connected component labels for each class
        gt_one_hot_cc: One-hot encoded tensor with connected component labels for each class
        metric: Function to calculate similarity between masks
        thing_cls_list: List of class indices to process (e.g., [1, 3])
        part_cls_list: List of part class indices (e.g., [2])
        
    Returns:
        Dictionary containing class-wise matches, similarity scores, and part component evaluations
    """
    # Initialize dictionaries to store results by class
    class_results = {}
    
    # Process only the classes in thing_cls_list
    if thing_cls_list is None:
        # Default to processing all classes
        num_classes = pred_one_hot_cc.shape[0]
        classes_to_process = range(1, num_classes)  # Skip background class (0)
    else:
        classes_to_process = thing_cls_list
    
    # Process each specified class
    for class_idx in classes_to_process:
        pred_to_gt = {}
        gt_to_pred = {}
        dice_scores = {}
        pred_parts = {}  # Map of pred component to list of part components within it
        gt_parts = {}    # Map of gt component to list of part components within it
        part_collective_scores = {}  # Collective scores for part components
        
        pred_class_cc = pred_one_hot_cc[class_idx]
        gt_class_cc = gt_one_hot_cc[class_idx]
        
        # Get unique component labels for this class
        pred_labels = torch.unique(pred_class_cc)[1:]  # Exclude background (0)
        gt_labels = torch.unique(gt_class_cc)[1:]  # Exclude background (0)
        
        # Create masks for each connected component
        pred_masks = {label.item(): pred_class_cc == label for label in pred_labels}
        gt_masks = {label.item(): gt_class_cc == label for label in gt_labels}
        
        # Match components and compute scores
        for (pred_item, pred_mask), (gt_item, gt_mask) in zip(pred_masks.items(), gt_masks.items()):
            if torch.any(torch.logical_and(pred_mask, gt_mask)):
                pred_to_gt.setdefault(pred_item, []).append(gt_item)
                gt_to_pred.setdefault(gt_item, []).append(pred_item)
                dice_scores[(pred_item, gt_item)] = metric(pred_mask, gt_mask)

                    
                # For each matching panoptic component pair, find part components inside them
                if part_cls_list:
                    # Dilate masks by one pixel for part detection
                    dilated_pred_mask = dilate_mask(pred_mask)
                    dilated_gt_mask = dilate_mask(gt_mask)
                    
                    # Store collective part masks for each part class
                    all_pred_part_masks = {}
                    all_gt_part_masks = {}
                    
                    # Check each part class
                    for part_cls in part_cls_list:
                        # Get part labels for this class
                        part_cc = pred_one_hot_cc[part_cls]
                        gt_part_cc = gt_one_hot_cc[part_cls]
                        
                        # Find parts within pred panoptic component (using dilated mask)
                        pred_part_items = []
                        pred_collective_mask = torch.zeros_like(pred_mask, dtype=torch.bool)
                        
                        for part_label in torch.unique(part_cc)[1:]:
                            part_mask = part_cc == part_label
                            # Check if part is within dilated pred panoptic component
                            if torch.any(torch.logical_and(dilated_pred_mask, part_mask)):
                                pred_part_items.append((part_cls, part_label.item()))
                                # Accumulate part masks for collective evaluation
                                pred_collective_mask = torch.logical_or(pred_collective_mask, part_mask)
                        
                        # Find parts within gt panoptic component (using dilated mask)
                        gt_part_items = []
                        gt_collective_mask = torch.zeros_like(gt_mask, dtype=torch.bool)
                        
                        for part_label in torch.unique(gt_part_cc)[1:]:
                            part_mask = gt_part_cc == part_label
                            # Check if part is within dilated gt panoptic component
                            if torch.any(torch.logical_and(dilated_gt_mask, part_mask)):
                                gt_part_items.append((part_cls, part_label.item()))
                                # Accumulate part masks for collective evaluation
                                gt_collective_mask = torch.logical_or(gt_collective_mask, part_mask)
                        
                        # Store part components for this panoptic pair
                        if pred_part_items:
                            pred_parts.setdefault((class_idx, pred_item), []).extend(pred_part_items)
                            all_pred_part_masks[part_cls] = pred_collective_mask
                        
                        if gt_part_items:
                            gt_parts.setdefault((class_idx, gt_item), []).extend(gt_part_items)
                            all_gt_part_masks[part_cls] = gt_collective_mask
                    
                    # Calculate collective part score for each part class
                    for part_cls in part_cls_list:
                        if part_cls in all_pred_part_masks and part_cls in all_gt_part_masks:
                            pred_parts_mask = all_pred_part_masks[part_cls]
                            gt_parts_mask = all_gt_part_masks[part_cls]
                            
                            # Compute collective score for this part class
                            collective_score = metric(pred_parts_mask, gt_parts_mask)
                            part_collective_scores[((class_idx, pred_item), (class_idx, gt_item), part_cls)] = collective_score
        
        # Ensure all labels are in dictionaries even if they have no matches
        for gt_label in gt_labels:
            gt_item = gt_label.item()
            gt_to_pred.setdefault(gt_item, [])
        for pred_label in pred_labels:
            pred_item = pred_label.item()
            pred_to_gt.setdefault(pred_item, [])
        
        # Store results for this class
        class_results[class_idx] = {
            "pred_to_gt": pred_to_gt,
            "gt_to_pred": gt_to_pred,
            "dice_scores": dice_scores,
            "pred_parts": pred_parts,
            "gt_parts": gt_parts,
            "part_collective_scores": part_collective_scores
        }
    
    return class_results

def get_all_matches(class_matches):
    """
    Get all matches from a specific class dictionary.
    
    Args:
        class_matches: Dictionary for a specific class from create_match_dict
        
    Returns:
        Tuple of (match_data, part_match_data)
    """
    match_data = []

    # Main matches
    for gt, preds in class_matches["gt_to_pred"].items():
        if not preds:
            match_data.append((None, gt, 0.0))
        else:
            for pred in preds:
                dice_score = class_matches["dice_scores"].get((pred, gt), 0.0)
                match_data.append((pred, gt, dice_score))

    for pred, gts in class_matches["pred_to_gt"].items():
        if not gts:
            match_data.append((pred, None, 0.0))

    # Part matches
    part_match_data = []
    for ((class_idx, pred), (class_idx_gt, gt), part_cls), score in class_matches["part_collective_scores"].items():
        part_match_data.append((pred, gt, part_cls, score))

    return match_data, part_match_data

def optimal_matching(match_data, part_matches=None):
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

    # Use linear_sum_assignment for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())

    optimal_matches = []
    for i, j in zip(row_ind, col_ind):
        pred = list(predictions)[i]
        gt = list(ground_truths)[j]
        score = 1 - cost_matrix[i, j].item()
        
        # Create base match information
        match_info = [pred, gt, score]
        
        # Add part scores if available
        if part_matches:
            part_scores = {}
            for p_pred, p_gt, part_cls, p_score in part_matches:
                if p_pred == pred and p_gt == gt:
                    part_scores[part_cls] = p_score.item() if hasattr(p_score, 'item') else p_score
            
            if part_scores:
                match_info.append(part_scores)
        
        optimal_matches.append(tuple(match_info))

    return optimal_matches

def thing_cls_counts(pred_one_hot_cc, gt_one_hot_cc, metric='dsc', thing_cls_list=None, part_cls_list=None):
    """
    Calculate panoptic counts (TP, FP, FN) for each class separately, restricted to thing_cls_list.
    
    Args:
        pred_one_hot_cc: Predicted connected components tensor
        gt_one_hot_cc: Ground truth connected components tensor
        metric: Metric name to use (default: 'dsc')
        thing_cls_list: List of panoptic class indices to process
        part_cls_list: List of part class indices (not used for panoptic counts)
        
    Returns:
        Dictionary with per-class counts {class_idx: (tp, fp, fn)}
    """
    if thing_cls_list is None:
        # Default to processing all classes except background
        thing_cls_list = range(1, pred_one_hot_cc.shape[0])

    class_counts = {}
    
    for cls in thing_cls_list:
        # Handle empty classes
        score, handled = _handle_empty_classes(pred_one_hot_cc[cls], gt_one_hot_cc[cls])
        if handled:
            tp = 1 if score == 1.0 else 0
            fp = 1 if score == 0.0 and torch.sum(pred_one_hot_cc[cls]) > 0 else 0
            fn = 1 if score == 0.0 and torch.sum(gt_one_hot_cc[cls]) > 0 else 0
            class_counts[cls] = (tp, fp, fn)
            continue

        # Create match dictionary for this class only - don't include part classes
        class_results = create_match_dict(pred_one_hot_cc, gt_one_hot_cc, 
                                         metric=METRIC_FUNCS[metric], 
                                         thing_cls_list=[cls],
                                         part_cls_list=None)  # Always None for panoptic counts
        
        if cls not in class_results:
            class_counts[cls] = (0, 0, 0)
            continue

        if part_cls_list is not None and cls in part_cls_list:
            # Skip part classes for panoptic counts
            continue
            
        # Get matches for this class
        match_data, _ = get_all_matches(class_results[cls])
        
        # Count false positives and false negatives
        fp = sum(1 for pred, gt, _ in match_data if gt is None)
        fn = sum(1 for pred, gt, _ in match_data if pred is None)
        
        # Get optimal matches for true positives
        optimal_matches = optimal_matching(match_data)
        tp = len(optimal_matches)
        
        # Store counts for this class
        class_counts[cls] = (tp, fp, fn)
    
    return class_counts, optimal_matches

def thing_cls_scores(optimal_matches, thing_cls_list=None, matching_dict=None):
    """
    Calculate quality score for each match considering both panoptic and part components.
    If thing_cls_list and matching_dict are provided, calculates scores across all classes.
    
    Args:
        optimal_matches: List of tuples of format (pred, gt, score) or (pred, gt, score, part_scores)
        thing_cls_list: Optional list of class indices to process
        matching_dict: Optional dictionary mapping class indices to matches
        
    Returns:
        If thing_cls_list and matching_dict are None:
            mean_quality: Mean quality score across all matches
            match_qualities: List of quality scores for each match
        Otherwise:
            mean_score: Mean score across all classes
            cls_wise_scores: Dictionary mapping class indices to their scores
    """
    # If we're just calculating scores for one set of optimal matches
    if thing_cls_list is None or matching_dict is None:
        match_qualities = []
        
        for match in optimal_matches:
            if len(match) == 4:  # Match has part components
                pred, gt, panoptic_score, part_scores = match
                # Calculate mean of panoptic score and all part scores
                all_scores = [panoptic_score] + list(part_scores.values())
                match_quality = sum(all_scores) / len(all_scores)
            else:  # Match doesn't have part components
                pred, gt, panoptic_score = match
                match_quality = panoptic_score
            
            match_qualities.append(match_quality)
        
        # Calculate mean across all matches
        # mean_quality = sum(match_qualities) / len(match_qualities) if match_qualities else 0
        total_score = sum(match_qualities)
        
        return total_score, match_qualities
    
    # If we're calculating scores across all classes
    else:
        cls_wise_scores = {}
        total_score = 0.0
        valid_classes = 0
        
        for class_idx in thing_cls_list:
            if class_idx in matching_dict:
                class_matches = matching_dict[class_idx]
                
                # Get all candidate matches for this class
                all_matches, part_matches = get_all_matches(class_matches)
                
                # Calculate optimal one-to-one assignment between predictions and ground truth
                optimal_matches = optimal_matching(all_matches, part_matches)
                
                if optimal_matches:  # Only calculate if we have matches
                    mean_cls_score, _ = thing_cls_scores(optimal_matches)
                    cls_wise_scores[class_idx] = mean_cls_score
                    total_score += mean_cls_score
                    valid_classes += 1
                else:
                    cls_wise_scores[class_idx] = 0.0
        
        # Calculate mean score across all classes
        # mean_score = total_score / valid_classes if valid_classes > 0 else 0.0
        
        return total_score, cls_wise_scores

def stuff_cls_scores(pred_one_hot_cc, gt_one_hot_cc, metric='dsc', stuff_cls_list=None):
    """
    Calculate quality score for stuff classes (semantic segments)
    
    Args:
        pred_one_hot_cc: Predicted connected components tensor
        gt_one_hot_cc: Ground truth connected components tensor
        metric: Metric name to use for comparison
        stuff_cls_list: List of stuff class indices to process
        
    Returns:
        mean_score: Mean quality score across all specified stuff classes
        
    Note:
        If stuff_cls_list is None or empty, returns 0
    """
    # Initialize total score
    total_score = 0.0
    
    # If no stuff classes provided, return 0
    if not stuff_cls_list:
        return 0.0
    
    # Calculate score for each stuff class and accumulate
    for cls_idx in stuff_cls_list:
        # Apply the specified metric to compare prediction with ground truth
        score = METRIC_FUNCS[metric](pred_one_hot_cc[cls_idx], gt_one_hot_cc[cls_idx])
        total_score += score
    
    # Calculate mean score across all stuff classes
    mean_score = total_score / len(stuff_cls_list)
    return mean_score

def PartPQ(pred_one_hot_cc, gt_one_hot_cc, metric='dsc', stuff_cls_list=None, thing_cls_list=None, part_cls_list=None):
    """
    Calculate Part-aware Panoptic Quality (PartPQ) scores
    
    Args:
        pred_one_hot_cc: Predicted connected components tensor
        gt_one_hot_cc: Ground truth connected components tensor
        metric: Metric name to use for comparison
        stuff_cls_list: List of semantic/stuff class indices
        thing_cls_list: List of instance/thing class indices
        part_cls_list: List of part class indices
        
    Returns:
        partpq_score: Overall PartPQ score combining instance and semantic quality
        
    Note:
        Handles both thing (instance) classes and stuff (semantic) classes
        Part classes are used for enhanced quality calculation but not counted directly
    """
    # Handle empty class lists
    thing_cls_list = thing_cls_list or []
    stuff_cls_list = stuff_cls_list or []
    part_cls_list = part_cls_list or []
    
    # Calculate instance counts and matches for thing classes
    counts, optimal_matches = thing_cls_counts(
        pred_one_hot_cc, gt_one_hot_cc, 
        metric=metric, 
        thing_cls_list=thing_cls_list, 
        part_cls_list=part_cls_list
    )
    
    # Filter out part classes from counts
    filtered_counts = {key: value for key, value in counts.items() if key not in part_cls_list}

    # Calculate total true positives, false positives, and false negatives
    total_tp = sum(tp for _, (tp, _, _) in filtered_counts.items())
    total_fp = sum(fp for _, (_, fp, _) in filtered_counts.items())
    total_fn = sum(fn for _, (_, _, fn) in filtered_counts.items())

    # Create match dictionary for quality calculation
    matching_dict = create_match_dict(
        pred_one_hot_cc, gt_one_hot_cc, 
        metric=METRIC_FUNCS[metric], 
        thing_cls_list=thing_cls_list, 
        part_cls_list=part_cls_list
    )
    
    # Get thing (instance) class quality score
    thing_cls_score, _ = thing_cls_scores(
        optimal_matches=optimal_matches, 
        thing_cls_list=thing_cls_list, 
        matching_dict=matching_dict
    )

    # Get stuff (semantic) class quality score
    stuff_cls_score = stuff_cls_scores(
        pred_one_hot_cc, gt_one_hot_cc, 
        metric=metric, 
        stuff_cls_list=stuff_cls_list
    )

    # Calculate final PartPQ score
    # Numerator: sum of thing and stuff quality scores
    # Denominator: total count of true positives, false positives, and false negatives
    numerator = thing_cls_score + stuff_cls_score
    denominator = total_tp + total_fp + total_fn + len(stuff_cls_list)

    # Avoid division by zero
    partpq_score = numerator / denominator if denominator > 0 else 0.0

    return partpq_score