# InstaMatch: Multi-Class Instance Sensitive Metrics for Semantic Segmentation. 

Like [panoptica](https://github.com/BrainLesion/panoptica/tree/main), this library is built for Computing instance-wise segmentation quality metrics for 2D and 3D semantic- and instance segmentation maps.  ***Our difference: Everything is on the GPU.*** We also have adopted all metrics for Multi-Class scenarios.


---
Developed by Soumya Snigdha Kundu, Tom Vercauteren, Aaron Kujawa, Marina Ivory Theodore Barfoot, and Jonathan Shapey.
___


### Instance Sensitive Metrics:

This is our personal nomenclature and naming to simplify stuff. We denote who initially proposed the actual logic in the links.

- One-to-One: Panoptic Metrics | [Kirillov et al.](https://arxiv.org/abs/1801.00868)
- Psuedo One-to-One: CC-Metrics | [Jaus et al.](https://arxiv.org/abs/2410.18684)
- Many-to-One: Lesion-wise Metrics | [BraTS-Mets Group](https://github.com/rachitsaluja/BraTS-2023-Metrics)
- Many-to-One: Maximise Merge Metrics | [Kofler et al.](https://arxiv.org/abs/2312.02608)
- Many-to-Many: Cluster Metrics | [Kundu et al.](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13407/1340711/Cluster-dice--a-simple-and-fast-approach-for-instance/10.1117/12.3047296.short)
- Many-to-Many: Multi Maximise Merge Metrics | Kundu et al. (Unpublished)
- Partial-to-One: Blob Metrics | [Kofler et al.](https://arxiv.org/abs/2205.08209)

#### Notes:

1. All are built for multiple classes out of the box.
2. For all `instance-sensitive-metrics`, the `base metrics` can be swapped out with a simple arg switch. Current Support:
   - Dice Score Coefficient.
   - Intersection over Union
   - Normalised Surface Distance. 

  We follow the normal pytorch convention of C, H, W, D. Based on user request, we can amend the library to work with B, C, H, W, D as well, like MONAI.

#### Terminology:

`instance/instance-sensitive-metric` -> Indicates a metric which is dependent on the indiviual scores of each components in a volume.

`semantic metric` -> Multiclass verisons of base metrics.

`base metric` -> Adapted semantic metrics to work with instance metrics easily.

### Usage

1. `git clone git@github.com:aymuos15/InstaMatch.git`

2. In your .py or .ipynb file: 

```python
from InstaMatch.instance_sensitive_metrics import panoptic_dice # Or any of the other 3.

score = panoptic_dice(pred, gt) # plain 2D/3D prediction and gt torch.tensor.
```

(or) You could simply copy and paste :P

For details and input dimensions: please open `unittest.ipynb`

### Similar Works: (Will be completed soon.)

Panoptica

- Limitations

Panoptic Quality Metric on MONAI

- Limitations:

### Contributions

We are always looking for contributions in any form.

### Guidelines
Final Guidelines coming soon ...

## Todo's Theory
- [ ] Decide the best way to approach empty pred/gt pairs.
- [ ] Mention section/line numbers of the paper when mentioned anywhere in the codebase. (Currently only mentioned in unit tests)


## Todo's Code
- [ ] Optimise code
  - [ ] All base/global metrics should be based on MONAI and optimised to use it. 
  - [ ] Mention properly in the docs/functions whether the gt or both gt and pred go through the connected components function.
- [ ] 3D Unit tests and corresponding visualisations.
- [ ] Add variable threshold options.
- [ ] [LOT OF WORK BUT IMPORTANT] - Make visualisations of the matching for all of the metrics.
- [ ] Make a requirements.txt
- [ ] All base and semantic metrics should be from MONAI

## Todo's immediate
- [ ] Understand the instance way for blob and region dice
- [ ] Do good docs and function names for all
- [ ] Make sure Normalised Surface Dice is properly implemented within Region Metrics
- [ ] Pass num classes as an arg.
- [ ] Properly/Elegantly handle cases where there is no prediction or ground truth. Throws errors at the moment.