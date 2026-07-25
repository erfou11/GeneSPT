# Protocol A public manifests

This directory mirrors the lightweight manifests stored in the data archive:

- `FORMAL_BASELINE_RUN_MANIFEST.csv`: 150 scheduler-managed
  external-method tasks.
- `STAI_FORMAL_ADOPTION_MANIFEST.json`: 30 stAI dataset-fold outputs,
  their raw prediction hashes, frozen test axes and centralized metric hashes.
- `INPUT_SPLIT_MANIFEST.csv`: 30 dataset-fold input and split contracts.
- `PREDICTION_MATRIX_MANIFEST.csv`: compact benchmark prediction matrices in
  the extracted archive. The release exporter regenerates this as 210 rows
  after adding the 30 stAI matrices.
- `PROTOCOL_A_TRUTH_MATRIX_MANIFEST.csv`: 30 fold-specific truth matrices.
- `MECHANISM_ABLATION_MATRIX_MANIFEST.csv`: 90 Figure 3A-C identity-readout
  mechanism matrices.
- `READOUT_SELECTION_MANIFEST.csv`: 60 GeneSPT/GeneSPT-GC selection rows.
- `BASELINE_SOURCE_MANIFEST.csv`: adapter and patched-runtime provenance.
- `DATASET_AUDIT_MANIFEST.csv`: dataset dimensions, formal method set and
  matrix counts.

All paths are archive- or repository-relative. Large arrays are distributed
through the Zenodo data archive.
