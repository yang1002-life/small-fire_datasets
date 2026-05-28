# Small-Scale Fire Datasets

This repository hosts the dataset resources associated with the paper **"FireDet: A framework for precise detection and localization of small-scale fire"**, published in *Fire Safety Journal*.

- Paper: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0379711226001050)
- DOI: [10.1016/j.firesaf.2026.104737](https://doi.org/10.1016/j.firesaf.2026.104737)
- Journal: *Fire Safety Journal*, Volume 162, Article 104737, 2026

## Overview

The work focuses on small-scale flame perception for early fire warning and localization. The dataset resources were built to support detection, generalization evaluation, and binocular localization experiments for small flame targets under diverse imaging conditions.

## Dataset summary

| Dataset | Scale | Purpose |
| --- | ---: | --- |
| Comprehensive Flame Dataset | 12,116 images | Model training and testing across diverse flame sizes and scenarios |
| Small-Target Flame Dataset | 9,209 images | Generalization evaluation for small-scale flame detection |
| Binocular Flame Localization Dataset | 1,498 stereo image pairs | Quantitative evaluation of long-distance flame localization |
| Calibration / optimization images | 839 checkerboard pairs + 120 flame pairs | Binocular camera calibration and parameter optimization |

The datasets include flame images collected from controlled experiments and diverse real-world-like scenes. The paper also integrates several public fire-related datasets for the comprehensive detection benchmark.

## Suggested structure


```text
small-fire_datasets/
├── comprehensive_flame_dataset/
│   ├── images/
│   └── labels/
├── small_target_flame_dataset/
│   ├── images/
│   └── labels/
├── binocular_localization_dataset/
│   ├── left/
│   ├── right/
│   └── annotations/
└── calibration_data/
    ├── checkerboard_left/
    ├── checkerboard_right/
    └── optimization_flame_pairs/
```

Labels are expected to follow the annotation format used in the experiments. If you use the data with YOLO-style detectors, please verify the class indices, image paths, and train/test split before training.

## Citation

If this dataset or paper is useful for your research, please cite:

```bibtex
@article{dai2026firedet,
  title   = {FireDet: A framework for precise detection and localization of small-scale fire},
  author  = {Dai, Jinyang and Cao, Qiuyang and Song, Xiaoning and Pan, Bence and Zhang, Qixing},
  journal = {Fire Safety Journal},
  volume  = {162},
  pages   = {104737},
  year    = {2026},
  doi     = {10.1016/j.firesaf.2026.104737}
}
```

## Notes

- Please check the paper for the full experimental setup, model design, and evaluation protocol.
- This repository is intended to provide dataset access and documentation for academic research.
- If dataset files are not yet visible in the repository, they still need to be uploaded separately.

## Contact

For questions about the paper or dataset, please refer to the author information in the published article.
