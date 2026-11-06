# Wind Power Forecasting: Do Time-Series Foundation Models Need Fine-Tuning?

**DATA612 PWS1 Group Project**: Deep Learning (DATA/MSML 612), Fall 2026, University of Maryland.

We forecast wind-farm power with a PatchTST transformer implemented from scratch, and compare it against the pretrained Chronos foundation model used zero-shot, fully fine-tuned, and fine-tuned with LoRA, each with 5%, 25% and 100% of the training data.

## Data

[SDWPF](https://arxiv.org/abs/2208.04360) (Baidu KDD Cup 2022): 134 turbines, 10-minute readings, 245 days. Download from Baidu AI Studio or the Hugging Face mirror `aigrids/WindFarm_raw`. Raw data is not committed to this repo.

Get the raw files with `scripts/download_sdwpf.sh` (checksummed, into `data/sdwpf/raw/`), then clean them with `python -m wpf.data.sdwpf`, which writes `data/sdwpf/clean.parquet` (plus `turbine_locations.parquet` beside it) and the counts in `results/ds01_counts.json`.

The split boundaries, training subsets and shared evaluation windows are fixed by `python -m wpf.data.splits --clean data/sdwpf/clean.parquet`, which writes `results/ds02_splits.json` (load it with `wpf.data.splits.load_splits`) and the coverage reports `results/ds02_window_coverage.json` and `results/ds02_window_counts.npz`.

The Chronos training subsets are exported as GluonTS Arrow files by `python -m wpf.data.chronos_arrow` (reads `data/sdwpf/clean.parquet`, writes `data/sdwpf/chronos/train_{5,25,100}pct.arrow`, `val.arrow` and `results/ds03_chronos_export.json`; `--flagged nan` blanks the unknown/abnormal readings instead of keeping them and writes to `data/sdwpf/chronos_nan/` and `results/ds03_chronos_export_nan.json`); it needs `gluonts` on Python 3.10-3.13, and `train.py` must be run with the `chronos_train_args` from that manifest.

## PatchTST on SDWPF

`python -m wpf.train_sdwpf --config configs/sdwpf_5pct.json --out output/sdwpf_5pct.json` trains one PatchTST shared across the 134 turbines on a training subset (`fraction` 0.05 / 0.25 / 1.0 in `configs/sdwpf_*.json`), with a masked loss over valid target steps, early stopping on the shared validation origins, and one scoring of the test origins at the end. The SDWPF configs set `quantiles` (0.1 to 0.9), so the model has a non-crossing quantile head trained with the pinball loss and the median is the point forecast; leave `quantiles` out for a point head trained with MSE. It writes the result JSON (with the shared DS-04 metrics for val and test), the best weights (`.pt`) and the kW predictions (`_pred.npz`: `val` and `test` medians of shape turbines x origins x 64, `val_q` and `test_q` with a leading quantile axis, and `quantiles`). `python -m wpf.score_sdwpf --pred output/sdwpf_5pct_pred.npz --name patchtst_5pct --out results/` scores such a file with `wpf.metrics` and writes `results/patchtst_5pct_val.json` and `_test.json`, the per-model result files. On Zaratan: `CFG=sdwpf_5pct sbatch slurm/sdwpf.sbatch` (`sdwpf_smoke` for a two-epoch check).

## Running on Zaratan

To run on UMD's Zaratan HPC cluster, see [docs/zaratan_setup.md](docs/zaratan_setup.md).

## References

- Nie et al. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023.
- Ansari et al. *Chronos: Learning the Language of Time Series.* TMLR 2024.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Zhou et al. *SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array.* Scientific Data 2024.

## Team

- Simran Kharbanda
- Tanmay Sharma
