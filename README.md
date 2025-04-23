# InstaMatch: Multi-Class Metrics for Instance-Sensitive Segmentation. 

This is a library to compute metrics for 2D/3D semantic, instance, panoptic and partaware segmentation. All on the GPU and multiple classes. The focus is on `instance sensitive` segmentation, where the metrics/results are dependent on the indiviual evaluation of each instance in a prediction or ground truth.

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

1. This library is predominantly posed to work with semantic segmentation outputs. However, it can be also be used for end to end instance, panoptic and part aware segmentation as well. Guidelines are in the making.
2. All metrics are built for multiple classes out of the box.
3. For all `instance` and `part` metrics, the `base metrics` can be swapped out with a simple arg switch. Current Support:
   - Dice Score Coefficient.
   - Intersection over Union
   - Normalised Surface Distance. 

  We follow the normal pytorch convention of C, H, W, D. Based on user request, we can amend the library to work with B, C, H, W, D as well, like MONAI.

#### Terminology:

`instance` -> Panoptic Quality style metrics which are more akin to panoptic or instance segmentation.

`semantic` -> Multiclass verisons of base metrics.

`part` -> Part aware segmentation metrics.

`base` -> Adapted semantic metrics to work with instance metrics easily.

### Usage (Will be completed soon.)

1. Clone the repo and `cd` into it.
2. `conda env create -f environment.yml`
3. `conda activate instamatch`

Unfortunately, I could not get a `pip` version working. Working on that (Help required as well :))

### Similar Works: (Will be completed soon.)

Panoptica

- Limitations

Panoptic Quality Metric on MONAI

- Limitations:

PanopticAPI

- Limitations:

### Contributions

We are always looking for contributions in any form. Immediately, we are looking to build exhaustive unit tests.

### Guidelines (Will be completed soon.)

...

## Todo's Theory
- [ ] Decide the best way to approach empty pred/gt pairs.
- [ ] Mention section/line numbers of the paper when mentioned anywhere in the codebase. (Currently only mentioned in unit tests)

## Todo's Code
- [ ] All semantic metrics should be from MONAI. (Or is this actually the best route?)
- [ ] Optimise code (If possible)
- [ ] 3D Unit tests.
  - [ ] Minor testing shows it works.
- [ ] Add variable threshold options.
- [ ] Pass num classes as an arg.
- [ ] [LOT OF WORK BUT IMPORTANT] - Make visualisations of the matching for all of the metrics.

## Part Metrics Todo's immediate
- [ ] Elabortate Documentation.
- [ ] Properly/Elegantly handle cases where there is no prediction or ground truth. Throws errors at the moment.
- [ ] Further optimisation.
- [ ] The plotting code in unit tests should not rely on the ordering of the classes defined.
- [ ] Need to set a threshold for PartPQ/PartRegion (This will go in `create_match_dict`).
- [ ] Deciding the best way to handle unit tests.