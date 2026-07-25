# Data and frozen splits

Large processed inputs, frozen splits, prediction matrices, and ground-truth
matrices are distributed through Zenodo:

- all versions: [10.5281/zenodo.21550226](https://doi.org/10.5281/zenodo.21550226)
- release v1.0.0: [10.5281/zenodo.21550227](https://doi.org/10.5281/zenodo.21550227)

The archive contains processed inputs, frozen gene splits, source tables,
prediction matrices, ground-truth matrices, label provenance, and SHA-256
manifests. Source datasets retain their original accession, citation, and
license terms.

Local experiment scripts use dataset-specific text or array layouts. The YAML
files in `configs/` document the intended dataset IDs and logical inputs. Do
not commit local copies of expression matrices to this repository.

Strict whole-gene holdout reserves test-gene ST expression for final evaluation
only. scRNA-derived descriptors for test genes are allowed as external gene
identity information and do not contain test-gene ST spatial expression.

stAI additionally requires author-provided scRNA cell labels for its
supervised reference encoder. Their source columns, alignment rules, cell
counts and source hashes are recorded in Supplementary Table S1 and the
archive label-provenance records. No pseudo-labels are used. Local full
retraining supplies these files through the six `STAI_*_LABEL_PATH`
environment variables documented in `baseline_adapters/stai/README.md`.
