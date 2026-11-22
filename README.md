# Aerial Small-Object Detection on VisDrone

Benchmarking YOLO detectors on drone imagery, where most objects are only a few pixels wide.
The question: **with a limited compute budget, does a bigger model or a bigger input image help more?**

> **Status: in progress.** Baseline and Run A are complete. Run B (higher resolution) is training. Results, code and plots are added as each experiment finishes.

## Dataset

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset): images captured from drones over urban and suburban scenes, 10 classes.

| Split | Images | Boxes |
|---|---|---|
| Train | 6,471 | 343,205 |
| Validation | 548 | 38,759 |

Data analysis of the training set (box sizes measured in original pixels, COCO size thresholds):

- **53 objects per image** on average, against about 7 in COCO: dense, overlapping scenes.
- **60.5% of objects are small** (< 32×32 px), 34.0% medium, 5.5% large. After resizing to 640 px for training, most objects become tiny.
- **Strong class imbalance:** 144,867 cars against 3,246 awning-tricycles (about 45×).

## Experiment design

Each run changes **one variable** against the baseline, so any change in results has a single cause.

| Run | Model | Input size | Tests |
|---|---|---|---|
| Baseline | YOLO11n (2.6M params) | 640 | Reference point |
| A | YOLO11s (~9.4M params) | 640 | Does more **model capacity** help? |
| B | YOLO11n (2.6M params) | 1280 | Does more **input resolution** help tiny objects? |

All runs: 30 epochs, COCO-pretrained weights, seed 0, Ultralytics 8.4, Kaggle Tesla T4.

## Results

| Model | Input | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| YOLO11n (baseline) | 640 | 0.414 | 0.315 | 0.299 | 0.165 |
| YOLO11s (Run A) | 640 | 0.511 | 0.377 | **0.380** | **0.218** |
| YOLO11n (Run B) | 1280 | *training* | | | |

## Findings so far

**Baseline error analysis** (confusion matrix and PR curves) shows two distinct failure modes:

1. **Missed small objects:** 42–44% of pedestrians, people and bicycles are never detected. Maximum recall for bicycle is about 0.5 at any confidence threshold, so the limit is what the model can see, not threshold choice.
2. **Confusion between similar vehicles:** 58% of vans are predicted as cars, a mix of class imbalance and small visual differences seen from above.

**Run A (bigger model):** mAP50 rose by 27% (0.299 → 0.380) and van mAP50 by 29% (0.345 → 0.444). Every class improved, not only the confused vehicles. Recall is still 0.377, so most small objects remain undetected. Run B tests whether resolution addresses this.

**Training notes:** the baseline was still improving at epoch 30 (not converged) with no sign of overfitting. Runs A and B use `deterministic=False` (seed fixed) because of a PyTorch/Triton version conflict in the Kaggle environment.

## Next steps

- [ ] Run B: YOLO11n at 1280 px
- [ ] Fair speed benchmark of all models (same GPU, batch size 1, ms per image)
- [ ] Zero-shot comparison with an open-vocabulary foundation model (YOLO-World)
- [ ] Sliced inference (SAHI) for small objects: accuracy gain vs latency cost
- [ ] Knowledge distillation from a larger teacher into YOLO11n
- [ ] ONNX export and FP16 for embedded deployment
- [ ] Future work: multi-object tracking on VisDrone video sequences

## Contact

Kishore Tambi Suddapalli · kishoresuddapalli8@gmail.com
