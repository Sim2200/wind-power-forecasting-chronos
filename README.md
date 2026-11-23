# Wind Power Forecasting: Do Time-Series Foundation Models Need Fine-Tuning?

**DATA612 PWS1 Group Project**: Deep Learning (DATA/MSML 612), Fall 2026, University of Maryland.

We forecast wind-farm power with a PatchTST transformer implemented from scratch, and compare it against the pretrained Chronos foundation model used zero-shot, fully fine-tuned, and fine-tuned with LoRA, each with 5%, 25% and 100% of the training data.

## Data

[SDWPF](https://arxiv.org/abs/2208.04360) (Baidu KDD Cup 2022): 134 turbines, 10-minute readings, 245 days. Download from Baidu AI Studio or the Hugging Face mirror `aigrids/WindFarm_raw`. Raw data is not committed to this repo.

Get the raw files with `scripts/download_sdwpf.sh` (checksummed, into `data/sdwpf/raw/`), then clean them with `python -m wpf.data.sdwpf`, which writes `data/sdwpf/clean.parquet` (plus `turbine_locations.parquet` beside it) and the counts in `results/ds01_counts.json`.

The split boundaries, training subsets and shared evaluation windows are fixed by `python -m wpf.data.splits --clean data/sdwpf/clean.parquet`, which writes `results/ds02_splits.json` (load it with `wpf.data.splits.load_splits`) and the coverage reports `results/ds02_window_coverage.json` and `results/ds02_window_counts.npz`.

The Chronos training subsets are exported as GluonTS Arrow files by `python -m wpf.data.chronos_arrow` (reads `data/sdwpf/clean.parquet`, writes `data/sdwpf/chronos/train_{5,25,100}pct.arrow`, `val.arrow` and `results/ds03_chronos_export.json`; `--flagged nan` blanks the unknown/abnormal readings instead of keeping them and writes to `data/sdwpf/chronos_nan/` and `results/ds03_chronos_export_nan.json`); it needs `gluonts` on Python 3.10-3.13, and `train.py` must be run with the `chronos_train_args` from that manifest.

## PatchTST on SDWPF

`python -m wpf.train_sdwpf --config configs/sdwpf_5pct.json --out output/sdwpf_5pct.json` trains one PatchTST shared across the 134 turbines on a training subset (`fraction` 0.05 / 0.25 / 1.0 in `configs/sdwpf_*.json`), with a masked loss over valid target steps, early stopping on the shared validation origins, and one scoring of the test origins at the end. It writes the result JSON (with the shared DS-04 metrics for val and test), the best weights (`.pt`) and the kW predictions (`_pred.npz`, `val` and `test` arrays of shape turbines x origins x 64). `python -m wpf.score_sdwpf --pred output/sdwpf_5pct_pred.npz --name patchtst_5pct --out results/` scores such a file with `wpf.metrics` and writes `results/patchtst_5pct_val.json` and `_test.json`, the per-model result files. On Zaratan: `CFG=sdwpf_5pct sbatch slurm/sdwpf.sbatch` (`sdwpf_smoke` for a two-epoch check). `configs/sdwpf_spatial_100pct_k4.json` (PT-05) is the grid recipe with `model_type: patchtst_spatial` and `neighbours: 4`: each window carries the centre turbine and its 4 nearest turbines (`wpf.data.spatial`, from `turbine_locations.parquet`) as channels, and `wpf.models.spatial_patchtst` puts all their patch tokens, with a turbine-slot embedding, in one encoder and forecasts the centre from its own tokens (`sdwpf_spatial_smoke` is the two-epoch check). `configs/sdwpf_point_*.json` are the same settings without `quantiles` and with the grid recipe (lr 3e-5 from PT-03), for the point-head ablation (MSE loss, WQL null) next to the main grid.

Head design (PT-03): `configs/sdwpf_100pct_h6.json` and `_h36.json` train one model per horizon on the full data; `python -m wpf.head_design --shared results/pt03_100pct.json --h6 results/pt03_100pct_h6.json --h36 results/pt03_100pct_h36.json` compares them with the shared 64-step head on validation WQL and writes the chosen design to `results/pt03_head_design.json`.

## LSTM baseline

`CFG=lstm_100pct_s2021 sbatch slurm/sdwpf.sbatch` trains the DS-06 LSTM (`model_type: lstm` in `configs/lstm_100pct_s{2021,2022,2023}.json`: RevIN, 2 x 128 LSTM over the 288-step window, the same quantile head, loss, early stopping and scoring as PatchTST) on the full training data for one seed; `lstm_smoke` is the two-epoch check. `python -m wpf.seed_summary results/lstm_100pct_s2021.json results/lstm_100pct_s2022.json results/lstm_100pct_s2023.json --out results/lstm_100pct_seeds.json` writes the mean and standard deviation over the seeds.

## Chronos zero-shot

`MODEL=amazon/chronos-t5-small NAME=chronos_small_zeroshot PERIODS="val test" sbatch slurm/chronos_eval.sbatch` scores the pretrained model on the shared windows without training: `python -m wpf.chronos_eval --model <hf id or checkpoint dir> --name <name> --out output/<name>.json [--results results/] [--periods val test] [--context keep|nan]` builds the 288-step kW contexts before the shared origins (missing readings as NaN, flagged readings as stored under `keep`), calls `predict_quantiles` for the 9 deciles, takes the median as the point forecast, writes the same `_pred.npz` layout as PatchTST and scores it with `wpf.metrics`. The model must be in the Hugging Face cache before the job, since compute nodes are offline. The same command scores a fine-tuned checkpoint directory.

## Chronos fine-tuning

`CFG=chronos_ft_5pct sbatch --gres=gpu:v100:1 slurm/chronos_ft.sbatch` fine-tunes `amazon/chronos-t5-small` on a DS-03 Arrow subset with Chronos' own `scripts/training/train.py` (`configs/chronos_ft_{5pct,5pct_nan,25pct,100pct}.yaml`: 1000 steps, lr 1e-3 annealed to 0, batch 32, context 288, prediction 64, min_past 288; `chronos_ft_smoke` is 50 steps of chronos-t5-tiny) and then scores `checkpoint-final` on the shared validation windows with `wpf.chronos_eval` (see Chronos zero-shot). The clone of chronos-forecasting is expected at `~/scratch.msml612/chronos-forecasting` (`CHRONOS_REPO` overrides). chronos-t5-small at batch 32 does not fit the 5 GB MIG slice; the V100 does a run in about 12 minutes. `configs/chronos_ft_5pct_lr{1e-4,1e-5}_nan.yaml`, `chronos_ft_{25pct,100pct}_lr1e-5_nan.yaml` and `chronos_ft_100pct_lr1e-5_nan_5k.yaml` are the recipe checks behind the grid's Chronos recipe (lr 1e-5, 1000 steps, nan export and contexts; see #52): the paper's lr 1e-3 damages the pretrained model on this data, and 5000 steps is worse than 1000. `configs/chronos_scratch_5pct_lr{1e-3,1e-5}_nan.yaml` (CF-05) train the same architecture from random initialization (`random_init: true`) with that recipe, the control showing the pipeline moves weights where there is something to learn (`results/chronos_scratch_*`).

`CFG=chronos_lora_5pct_lr1e-4_nan RANK=8 sbatch slurm/chronos_lora.sbatch` fine-tunes the same model with LoRA instead: `scripts/train_lora.py` runs Chronos' `train.py` with the same YAML config but wraps the T5 attention projections (q, k, v, o) with `peft` adapters of rank `RANK` (alpha 2 x rank, dropout 0.05; `wpf.lora`), so only the adapters train, then merges them into the weights and saves a plain `checkpoint-final` that `wpf.chronos_eval` scores unchanged; `run-N/adapter` keeps the adapter alone and `checkpoint-final/lora.json` (recorded as `train_info`) the rank and trainable-parameter count. `configs/chronos_lora_5pct_lr{1e-4,1e-5}_nan.yaml` are the CF-01 recipe at 10x and 1x the full fine-tuning rate, `chronos_lora_5pct_lr1e-5_nan_s{2022,2023}.yaml` the recipe on two more seeds (`results/chronos_lora_5pct_lr1e-5_nan_r8_seeds.json` is the three-seed summary) and `RANK=4|16|32` on the lr 1e-5 config the rank sweep of #59 (`results/chronos_lora_5pct_lr1e-5_nan_r{4,8,16,32}*.json`; `python scripts/plot_lora_rank_sweep.py` draws `docs/figures/lora_rank_sweep.png`, MASE and WQL against trainable parameters, with matplotlib); `chronos_lora_smoke` is 50 steps of chronos-t5-tiny. The adapters fit the MIG slice.

`CFG=chronos_ft_base_5pct_nan GPUS=2 sbatch --gres=gpu:v100:2 slurm/chronos_ft_ddp.sbatch` fine-tunes `amazon/chronos-t5-base` (200M parameters) on 2 V100s with DDP (CF-03): `torchrun --nproc_per_node $GPUS` around the same `train.py`, whose Hugging Face Trainer shards the batches across the processes; the config's `per_device_train_batch_size` (32) is read as the effective batch and passed as 32 / GPUS per GPU, so the effective batch and the 1000 optimizer steps are the same on any GPU count and the config is the CF-01 grid recipe with the base model (`tests/test_chronos_base_configs.py`; V100 has no TF32 or bf16, so training is fp32 like the small-model runs). `chronos_ft_base_scaling` is the same at 200 steps for timing (`EVAL=0`, `GPUS=1|2|4`); `results/cf03_scaling.json` records wall time, steps per second and SU per GPU count. `MODEL=amazon/chronos-t5-base NAME=chronos_base_zeroshot_nan CONTEXT=nan sbatch slurm/chronos_eval.sbatch` is the base model's zero-shot reference.

## The experiment grid

`python -m wpf.grid submit` launches the whole grid (EX-01) as one Slurm job array on the V100s, one GPU type for every cell so the GPU hours of the cost analysis are comparable across models: {PatchTST quantile, PatchTST point, LSTM, Chronos full fine-tune, Chronos LoRA, Chronos zero-shot}, each cell being {5, 25, 100%} x {seeds 2021, 2022, 2023} with the recipe its own task settled on (`wpf.grid.MODELS`). Each array task runs the model's existing sbatch script inside its allocation with `CFG`, `SEED`, `NAME=ex_<model>_<pct>pct_s<seed>` and `PERIODS="val test"`, so every run scores validation and, exactly once, the test windows, and its record `output/<name>.json` carries `train_seconds`, `inference_seconds`, `gpu` and the parameter count. `python -m wpf.grid list` is the dry run (one line per cell), `status` shows which cells have a record, and `submit --missing` re-submits only the cells without one. `python -m wpf.grid collect` (EX-02) copies every cell's record to `results/ex_<name>.json`, writes the three-seed summary of each (model, fraction) to `results/ex_<model>_<pct>pct_seeds.json` with `wpf.seed_summary`, and `results/ex_grid.json`: one row per cell with its metrics, GPU, Slurm job id, train and inference seconds and, from `sacct`, elapsed time and billed SU, plus the DS-05 persistence and seasonal-naive scores as `baselines` rows so the analysis reads one table (`--no-sacct` skips the accounting, `--job NAME=JOBID` supplies an id a record lacks). The grid of 2026-11-19 to 2026-11-23 cost 107.8 SU for 46 cells.

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
