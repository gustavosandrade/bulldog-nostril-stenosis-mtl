# Bulldog Nostril Stenosis - Multi-Task Learning

Image-free reproducibility package for Access-2026-50300
Multi-Task Learning With Auxiliary Segmentation for Visual Grading of Nostril Stenosis in Bulldogs

Package version: 1.0.0
Version tag: v1.0.0
Repository: https://github.com/gustavosandrade/bulldog-nostril-stenosis-mtl

This standalone repository contains only the reviewed image-free package.
It does not include the private research repository or its commit history.
No DOI is claimed. No separate code/data reuse license is granted with this
release; public access does not imply a CC BY or other open-source license.
The article's publication license is separate from any repository license.

INVENTORY
manifests/fold_assignments.csv: 10,200 assignment rows for the frozen seed-42
bundle, covering internal training, real internal validation, real external
testing and excluded synthetic groups. Each real crop is externally tested
once. Sample IDs and group IDs are study identifiers, not owner identities.
Source image/mask paths and personal directories have been removed.
Some original sample identifiers and target-construction statistics retain
the label "open". The frozen three-class experiment maps open to mild;
the fold class_name and prediction true_class fields contain that mapping.

predictions/oof_predictions.csv: 2,280 original saved predictions, comprising
12 conditions x 190 real nostril crops from 95 dogs. Original probabilities
are preserved at their stored precision. There are no generated observations.
predictions/fold_metrics.csv: 120 archived fold-level macro-F1 values.
predictions/archived_summary.json: full-precision fold summaries and confusion
matrices. Fold mean and pooled out-of-fold metrics are different estimators.

target_generation/: the three original mask-construction scripts and their
two helper modules, copied without algorithm changes. These require the
original input masks/manifests, which are not redistributed here. The shrink
strategy is '--strategy shrink', not the script's default intensity strategy.
target_statistics/: per-target construction statistics without file paths.
recompute_summaries.py: standard-library-only validation and recalculation
from the saved predictions. It performs no training or image reconstruction.
PROVENANCE.json: source fingerprints and package identification.

REPRODUCING THE NUMERICAL SUMMARIES
From this directory, run:
    python3 recompute_summaries.py
The command writes derived_checks.json, checks complete coverage and dog-level
separation, and recomputes pooled confusion matrices, macro-F1, class-wise F1,
ordinal MAE, extreme-error rate, and the exact paired signed-rank/Holm results.
Archived fold scores are used for the signed-rank test, retaining their original
precision instead of substituting rounded manuscript values.

MASK CONSTRUCTION WITH AUTHORIZED SOURCE DATA
The original commands accept --manifest, --data-root and --output-root. Run
build_corrected_dataset.py on the original masks; then refine_anatomic_masks.py
with --strategy shrink on the corrected outputs; finally run
build_geometric_mask_control.py on the shrink manifests. Use separate empty
output directories; do not use --force on research data. Dependencies: NumPy,
Pillow and OpenCV for mask construction. The source version's requirements.txt
does not pin every historical dependency. This package does not claim a fully
recovered, bitwise-identical historical training environment. Original training
used PyTorch 2.8/CUDA; the previously reported illustrative inference used
PyTorch 2.13/CPU. No new training has been performed for these final edits.

IMAGE ACCESS AND LIMITATIONS
No real/synthetic images, original masks, model weights, animal-owner records,
credentials or author portraits are included. Image access remains subject to
the original consent, ownership and institutional conditions (UCDB protocol
010/2022). Requests may be directed to the article's corresponding author;
release is not guaranteed by this package. Anonymous image-free summaries do
not confer rights to redistribute the original collection. The complete image
training pipeline cannot be rerun from this image-free package alone.

INTERPRETATION
The final comparisons reuse one cohort and one grouped partition. Training
folds overlap, and lambda selection reused these animals. The corrected-mask
ResNet18 gain is exploratory and does not establish independent replication,
anatomical mask accuracy or clinical utility. Preserve these limitations when
reusing the results.
