from metrics.base.dice_score_coefficient import dsc
from metrics.base.intersection_over_union import iou
from metrics.base.normalised_surface_dice import nsd

METRIC_FUNCS = {
    'dsc': dsc,
    'iou': iou,
    'nsd': nsd,
}

def _handle_empty_classes(pred_cls, gt_cls):
    """Helper function to handle empty class cases."""
    if pred_cls.sum() == 0 and gt_cls.sum() == 0:
        return 1.0, True  # Score 1.0, class handled
    elif pred_cls.sum() == 0 and gt_cls.sum() > 0:
        return 0.0, True  # Score 0.0, class handled
    elif gt_cls.sum() == 0 and pred_cls.sum() > 0:
        return 0.0, True  # Score 0.0, class handled
    
    return 0.0, False  # Class not handled, needs further processing

def _calculate_final_score(total_score, num_classes):
    """Calculate final score across classes excluding background."""
    return total_score / (num_classes - 1)