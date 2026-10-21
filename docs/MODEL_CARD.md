# Model card: lungxray EfficientNet-B0 pneumonia classifier

Binary classifier for pediatric chest X-rays (NORMAL vs PNEUMONIA) with a
temperature-scaled probability, a validation-tuned decision threshold and Grad-CAM
heatmaps. **Research and education only; not a medical device.**

| | |
| --- | --- |
| Architecture | EfficientNet-B0 (torchvision), ImageNet-pretrained, new 2-class head |
| Input | 224x224 center crop after resizing the shorter side to 256; greyscale repeated to 3 channels; ImageNet normalization |
| Output | P(pneumonia) after temperature scaling, label by threshold, borderline flag, Grad-CAM map |
| Artifact | `best.pt` bundle: weights plus preprocessing, threshold, temperature, training config, git commit, dataset fingerprint and metrics |
| Code | this repository, MIT licence |

## Intended use

- Teaching and research on medical image classification, calibration, threshold
  selection and explanation methods.
- Benchmarking and reproducing the pipeline on the same public dataset.

## Out-of-scope use

- Any clinical use: diagnosis, triage, screening, treatment decisions or second opinions.
- Adults, or children outside the training population (see below).
- Images that are not frontal chest radiographs, or findings other than pneumonia. The
  model always answers NORMAL or PNEUMONIA, even for unrelated images.
- Distinguishing bacterial from viral pneumonia (the labels were merged).

## Training data

Pediatric chest X-rays from Kermany et al. (2018): anterior-posterior images from
retrospective cohorts of children aged one to five at Guangzhou Women and Children's
Medical Center, acquired during routine care and graded by expert physicians. Licence
CC BY 4.0 ([doi:10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2)).

The copy used here holds the same 5,856 images as the public release, split differently:
train 1266 NORMAL / 3418 PNEUMONIA, validation 158 / 427, test 159 / 428. The splits were
used as provided; patient ids parsed from the file names show that many patients appear in
more than one split (see the overlap note below). The 4,684 training images are JPEGs with
widths of 384–2890 and heights of 127–2713 pixels (median 1280x888).

Training: two stages with AdamW and class-weighted cross-entropy. Stage 1 trains the head
on a frozen backbone for 3 epochs (learning rate 1e-3); stage 2 unfreezes the last 3
feature blocks for up to 15 epochs (backbone 1e-4, head 1e-3, 1 warmup epoch then cosine
decay), with mixed precision, gradient clipping at 1.0 and early stopping on validation
AUROC (patience 5). Augmentation: random resized crop (85–100% of the area), rotation up
to 7 degrees, 5% translation, brightness and contrast jitter of 0.15; no horizontal flip.
All settings: `configs/default.yaml`. In the reported run the best validation AUROC came
at the last stage-2 epoch, so early stopping did not trigger.

## Evaluation data

The test split of the same copy, used once with the threshold and temperature frozen from
validation. Because patients overlap between splits, the report also gives metrics for the
test images whose patient never appears in training. There is no external test set.

## Metrics

<!-- BEGIN GENERATED: model-card -->
### Test-set metrics

Test split: **587 images** (159 NORMAL, 428 PNEUMONIA). Decision threshold **0.938** on temperature-scaled P(pneumonia) (T = 2.40); both were fitted on the validation split, never on test. 95% CIs in parentheses: percentile bootstrap, 1000 resamples, seed 42. ECE uses 15 equal-width bins on P(pneumonia).

| Metric | All test images (n = 587) | Test patients not in train (n = 240) |
| --- | --- | --- |
| AUROC | 0.988 (0.976–0.996) | 0.979 (0.963–0.992) |
| AUPRC | 0.994 (0.987–0.998) | 0.963 (0.925–0.989) |
| Accuracy | 92.7% (90.5–94.7) | 91.2% (87.5–94.6) |
| Sensitivity (recall) | 91.4% (88.8–94.0) | 84.0% (75.9–91.0) |
| Specificity | 96.2% (92.7–98.8) | 95.9% (92.3–98.7) |
| PPV (precision) | 98.5% (97.1–99.5) | 92.9% (86.9–97.8) |
| NPV | 80.5% (74.9–86.1) | 90.3% (85.4–94.5) |
| F1 | 0.948 (0.932–0.962) | 0.883 (0.829–0.928) |
| ECE | 0.047 (0.038–0.065) | 0.122 (0.097–0.158) |

> **Patient overlap.** 347 of the 587 test images come from the 299 patients who also appear in the training split (the dataset folders were used as provided; see [Limitations](#caveats-and-limitations)), so the left column is optimistic. The right column keeps only test patients never seen in training (146 NORMAL, 94 PNEUMONIA); its class mix differs, so PPV and NPV are not comparable across the columns.

Confusion matrix (all test images): 391 true positives, 37 false negatives (missed pneumonia), 6 false positives, 153 true negatives.

### Validation and operating point

Validation split (585 images; used only for model selection, the temperature and the threshold): the threshold is the highest reaching 95% sensitivity there (95.1%, specificity 96.8%; Youden-J threshold 0.902). Validation AUROC 0.994.

### Calibration

Temperature scaling (T = 2.40) changed test ECE from 0.041 to 0.047; on validation it lowered NLL from 0.192 to 0.131 while ECE went from 0.046 to 0.052. Temperature scaling minimizes NLL, not ECE, so treat the displayed probability as approximate. See `reports/reliability.png`.

<sub>Generated by `scripts/render_results.py` from `reports/metrics.json`: efficientnet_b0 trained at commit e6f1605, evaluated at commit d43fe7d on 2026-10-20, dataset fingerprint `2f402612cb0c6f1d`.</sub>
<!-- END GENERATED: model-card -->

Plots: `reports/confusion_matrix.png`, `reports/roc.png`, `reports/pr.png`,
`reports/reliability.png`, `reports/gradcam_examples.png`.

## Ethical considerations

- **Risk of harm.** A missed pneumonia (false negative) can delay care; a false positive
  can cause unnecessary tests or treatment. The tool must not be put in a position to make
  either decision.
- **Automation bias.** A confident score and a heatmap can look more authoritative than
  they are. The app shows the uncertainty (borderline band, approximate calibration) and a
  persistent disclaimer.
- **Population bias.** One hospital, one country, one age band. Performance on other
  groups is unknown and may be much worse.
- **Privacy.** The app and API process uploads in memory, keep no images, and log only a
  hash of each file (no file names). The hosting provider's own logging still applies.
  Do not upload identifiable images to a public deployment.

## Caveats and limitations

- Shared patients between splits make the test metrics optimistic; the unseen-patient
  subset is the better guide but is small.
- The validation-tuned threshold did not reach its sensitivity target on test.
- Temperature scaling lowered validation NLL but did not lower ECE.
- Grad-CAM is a coarse 7x7 map of what raised the predicted class's score; it is not a
  localization method. Some images carry side markers or text that the model may exploit.
- The input check flags blank, colourful and tiny images only; it is not an X-ray
  detector.
- Retraining on a GPU is seeded but not bit-for-bit deterministic; evaluating a given
  bundle is.
