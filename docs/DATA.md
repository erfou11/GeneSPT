# Data and frozen splits

Large matrices and prediction files are not tracked in GitHub.

- DOI: <https://doi.org/10.5281/zenodo.21223023>
- Record: <https://zenodo.org/records/21223023>

The archive contains reviewer-facing processed inputs, frozen gene splits,
source tables, prediction matrices, ground-truth matrices, label provenance,
and SHA-256 manifests. Raw source datasets retain their original accession,
citation, and license terms.

Local experiment scripts use dataset-specific text or array layouts. The YAML
files in `configs/` document the intended dataset IDs and logical inputs. Do
not commit local copies of expression matrices to this repository.

Strict whole-gene holdout reserves test-gene ST expression for final evaluation
only. scRNA-derived descriptors for test genes are allowed as external gene
identity information and do not contain test-gene ST spatial expression.
