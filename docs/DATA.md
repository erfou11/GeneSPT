# Data and frozen splits

Large processed inputs, frozen splits, prediction matrices, and ground-truth
matrices are distributed through Zenodo:

- all versions: [10.5281/zenodo.21550226](https://doi.org/10.5281/zenodo.21550226)
- release v1.0.0: [10.5281/zenodo.21550227](https://doi.org/10.5281/zenodo.21550227)

The archive contains processed inputs, frozen gene splits, source tables,
prediction matrices, ground-truth matrices, label provenance, and SHA-256
manifests. Source datasets retain their original accession, citation, and
license terms.

## Original public sources

The fixed matrices used in this release are identified by their SHA-256 values
in the Zenodo archive. The original study/source routes are:

| Dataset | Original source and exact route |
| --- | --- |
| HBC | Wu et al. human breast-cancer atlas: [Zenodo 4739739](https://zenodo.org/records/4739739), DOI `10.5281/zenodo.4739739`, section `1142243F`, using `filtered_count_matrices.tar.gz` and `metadata.tar.gz`; paired scRNA-seq: [GEO GSE176078](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078). |
| Cell2location mouse brain | Kleshchevnikov et al., DOI `10.1038/s41587-021-01139-4`; [E-MTAB-11114](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11114) Visium and [E-MTAB-11115](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11115) paired snRNA-seq, released 2021-11-12. The spatial sample is `ST8059048` (`filtered_feature_bc_matrix.h5`). |
| MHPR/MERFISH | Moffitt et al., DOI `10.1126/science.aau5324`; [Dryad 10.5061/dryad.8t8s248](https://datadryad.org/dataset/doi:10.5061/dryad.8t8s248), exact file `Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv` (MD5 `25a51abdf981039949cfdaf4db0a9ab3`); companion 10X scRNA-seq: [GEO GSE113576](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113576). The source-side raw filename for the fixed 31,299-cell scRNA matrix was not retained in the historical benchmark download. |
| MVC/STARmap | Wang et al., DOI `10.1126/science.aat5691`; [Zenodo 10698912](https://zenodo.org/records/10698912), DOI `10.5281/zenodo.10698912`, exact file `STARmap_mouse_visual_cortex.zip` containing `STARmap_20180505_BY3_1k.h5ad`; Smart-seq reference: Tasic et al., DOI `10.1038/s41586-018-0654-5`, [GEO GSE115746](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746). |

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
