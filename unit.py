import torch
from metrics.instance.region import cc_metric
from metrics.tools.labelling import label
from metrics.tools.one_hot_encoding import one_hot_encode

# Sample Data
pred = torch.zeros((30, 30)).cuda()
gt = torch.zeros((30, 30)).cuda()

# Class 1 - Both pred and gt have the same region
pred[1:5, 1:5] = 1
gt[1:5, 1:5] = 1

# Class 1 - Both pred and gt have the same region
pred[7:11, 7:11] = 1
gt[7:11, 7:11] = 1

# Class 2 - Different regions in pred and gt
pred[25:32, 5:10] = 1
gt[25:32, 7:12] = 1  # Slight offset for partial overlap

pred = one_hot_encode(pred, num_classes=2)
gt = label(gt, num_classes=2)

print(cc_metric(pred, gt, metric='dsc'))  # Expected output: 0.5
# print(cc_metric(pred, gt, metric='iou'))  # Expected output: 0.5
# print(cc_metric(pred, gt, metric='nsd'))  # Expected output: 0.5