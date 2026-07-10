# Reproduction status

## Available

- Cleaned active GeneSPT and PSP source files.
- Dataset configuration templates.
- Lightweight metric/I/O package and synthetic tests.
- Manuscript figure-generation scripts retained by the publication audit.
- Zenodo DOI and output-schema documentation.

## External by design

- Raw and processed biological matrices.
- Frozen binary split arrays.
- Saved prediction matrices and checkpoints.
- Complete third-party baseline repositories.

## Current guarded experiment

`main/run_predictable_spatial_program_folds012.py` defaults to folds 0-4 and
the canonical `pca32_nmf32` PSP descriptor. It requires an explicitly supplied
canonical GC cache for reuse. Training a replacement base or selecting a
non-canonical descriptor requires an explicit override and must not be confused
with manuscript evidence.
