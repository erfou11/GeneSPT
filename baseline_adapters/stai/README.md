# stAI Protocol A adapter

`run_stai_protocol_a.py` is the exact three-stage wrapper used to add stAI to
the formal seven-method benchmark. It imports the official stAI model at commit
`3376cc16cc6d8461edafc0aeb4519b92d18474b7`.

```bash
git clone https://github.com/gszou99/stAI.git ../stAI
git -C ../stAI checkout 3376cc16cc6d8461edafc0aeb4519b92d18474b7
```

The reported run used 500 epochs, five internal models, `topk=50`,
`spatial_knn=10`, seed 8848, and the model/loss defaults recorded in
`configs/protocol_a_baseline_versions.yaml`.

Set `GENESPT_ARCHIVE_ROOT` to the extracted Zenodo archive and
`GENESPT_PROTOCOL_INPUT_ROOT` to the directory containing the frozen
`mode_a_split.json` and `full_truth.npy` files. Set `STAI_ROOT` when the
official checkout is not adjacent to this repository. The six optional
`STAI_*_LABEL_PATH` variables identify the author-provided reference-cell
annotations listed in `docs/BASELINE_ADAPTATION.md`.

Each stage is a separate process. The `prepare` and `run` stages cannot read
final-test ST expression; only `evaluate` opens the frozen truth.

```bash
python baseline_adapters/stai/run_stai_protocol_a.py prepare \
  --dataset Vis9A --fold 0 --output-dir results/stai/Vis9A/fold0

python baseline_adapters/stai/run_stai_protocol_a.py run \
  --dataset Vis9A --fold 0 --output-dir results/stai/Vis9A/fold0

python baseline_adapters/stai/run_stai_protocol_a.py evaluate \
  --dataset Vis9A --fold 0 --output-dir results/stai/Vis9A/fold0
```

Valid dataset keys are `Vis9A`, `HBC`, `Cell2location`, `seqFISH+`, `MHPR`,
and `MVC`.
