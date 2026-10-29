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

## Comprehensive Flame Dataset validation split download

The original validation split of the **Comprehensive Flame Dataset** (`fire_rgb`) is available on Hugging Face:

**[yang1002/comprehensive-flame-dataset-validation](https://huggingface.co/datasets/yang1002/comprehensive-flame-dataset-validation)**

This release contains **2,424 validation images** (approximately 2.14 GB including annotations), 2,424 YOLO TXT labels and 2,424 original XML annotations. These counts describe this validation release and are separate from the dataset totals above. No training split is included.

- YOLO labels use the single class `0: fire`, with 3,231 fire bounding boxes.
- The 899 empty TXT files are retained: they indicate no fire boxes in the supplied labels, and may still correspond to smoke-containing images.
- Original XML annotations contain the same 3,231 fire boxes plus 576 smoke boxes. TXT and XML therefore have different class coverage.
- Original image, label and XML files are preserved. `manifest.json` and `SHA256SUMS` provide file integrity checks.
- No dataset license has been specified. Do not infer one from the project code or dependencies; consult the dataset owner regarding reuse.

Download with the Hugging Face CLI:

```bash
pip install --upgrade huggingface_hub
hf download yang1002/comprehensive-flame-dataset-validation --repo-type dataset --local-dir comprehensive-flame-dataset-validation
```

The download preserves `images/val`, `labels/val` and `annotations/val`. See the [dataset card](https://huggingface.co/datasets/yang1002/comprehensive-flame-dataset-validation) for annotation details, Python download instructions and reproducible downloads using a fixed commit revision.

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
- The Comprehensive Flame Dataset (`fire_rgb`) validation split is hosted on Hugging Face via the link above; other dataset resources are not included in that release.

## Contact

For questions about the paper or dataset, please refer to the author information in the published article.
