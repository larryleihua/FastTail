# Build-time only

Nothing here ships in the installed package (`data-raw/` is in `.Rbuildignore`).
Users never touch these files.

- `nbe/*.bson` — the original Julia/Flux checkpoints, kept for provenance.
- `export_weights.py` — parses the BSON, writes `weights.bin` + `weights_index.csv`.
- `build_sysdata.R` — reshapes those into `../R/sysdata.rda`.

To regenerate after retraining, from the repo root:

```sh
pip install pymongo numpy
python3 data-raw/export_weights.py
Rscript data-raw/build_sysdata.R
```

The two intermediates are gitignored. Commit the refreshed `R/sysdata.rda`.
