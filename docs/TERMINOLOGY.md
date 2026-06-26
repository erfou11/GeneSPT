# Manuscript Terminology Notes

This repository preserves selected final local manuscript scripts in `main/`
and `scripts/`. Some of those anchored files retain historical internal labels
or path names. Public-facing documentation, figure labels, and manuscript text
should use the terminology below.

## Public Manuscript Terms

- Main model: `GeneSPT`
- Gene-conditioned baseline: `GeneSPT-GC`
- Spatial program component: `PSP` / Predictable Spatial Program Transfer
- Baseline display name: `TransImp`
- Primary datasets: Vis9A, HBC, Cell2location mouse brain
- Cross-platform datasets: seqFISH+ cortex/SVZ, MHPR/MERFISH, MVC/STARmap
- Metrics: SPCC, RMSE, JS/JSD, SSIM

## Preserved Internal Labels

These labels may appear inside anchored scripts or historical file paths:

- `TransPA`: preserved as an internal source-table/path key; display mappings
  convert it to `TransImp` where figure-facing labels are generated.
- `GeneSPT-GC-PSP`, `TopoDiST-GC-PSP`, or `current_descriptor_psp`: preserved
  internal names for the final GeneSPT readout path.
- `MHM`: preserved only inside anchored historical scripts that were copied
  from the final local source tree. It is not a current manuscript primary or
  cross-platform dataset.
- `GeneSPT-LCR` or `LCR`: preserved only where old anchored scripts document
  historical display filtering or comments. It is not part of the public
  manuscript method set.
- `raw SSIM`: preserved inside source-table or figure-generation code to
  distinguish unscaled SSIM from older SSIMx10 variants. Public metric naming
  should be `SSIM`.

## Release Rule

Do not introduce new public-facing references to deprecated variants such as
`GeneSPT-LCR`, `LCR`, or `MHM`. If an anchored script must be edited for public
execution, prefer a thin wrapper or documentation note unless changing the
label is clearly non-behavioral.
