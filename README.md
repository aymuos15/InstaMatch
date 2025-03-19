# InstaMatch: Multi-Class Instance Sensitive Metrics for Semantic Segmentation. 

Like [panoptica](https://github.com/BrainLesion/panoptica/tree/main), this library is built for Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.

---
Developed by Soumya Snigdha Kundu, Tom Vercauteren, Aaron Kujawa, Jonathan Shapey, and Marina Ivory.
___

 ***Our difference: Everything is on the GPU.*** 

### Instance Sensitive Metrics:

- One-to-One: Panoptic Quality | [Kirillov et al.](https://arxiv.org/abs/1801.00868)
- Psuedo One-to-One: CC-Metrics | [Jaus et al.](https://arxiv.org/abs/2410.18684)
- Many-to-One: Lesion-wise Dice | [BraTS-Mets Group](https://github.com/rachitsaluja/BraTS-2023-Metrics)
- Many-to-Many: Cluster Dice | Kundu et al. (To appear in SPIE Medical Imaging)
<!-- - Partial-to-One: Blob Dice | [Kofler et al.](https://arxiv.org/abs/2205.08209) -->

#### Notes:

1. All are built for multiple classes out of the box.
2. For all `instance-sensitive-metrics`, the `base metrics` can be swapped out with a simple arg switch. Current Support:
   - Dice Score Coefficient.
   - Intersection over Union
   - Normalised Surface Distance. 

#### Terminology:

`instance-sensitive-metric` -> Indicates a metric which is dependent on the indiviual scores of each components in a volume.

`base metric` -> regular metric like DSC, IoU etc.

### Usage

1. `git clone git@github.com:aymuos15/InstaMatch.git`

2. In your .py or .ipynb file: 

```python
from InstaMatch.instance_sensitive_metrics import panoptic_dice # Or any of the other 3.

score = panoptic_dice(pred, gt) # plain 2D/3D prediction and gt torch.tensor.
```

(or) You could simply copy and paste :P

For details and input dimensions: please open `test.ipynb`

### Contributions

We are always looking for scripts to stress test all the metrics on various edge cases. Few of which can be found in `test.ipynb` .