# Wind Power Forecasting: Do Time-Series Foundation Models Need Fine-Tuning?

**DATA612 PWS1 Group Project**: Deep Learning (DATA/MSML 612), Fall 2026, University of Maryland.

We forecast wind-farm power with a PatchTST transformer implemented from scratch, and compare it against the pretrained Chronos foundation model used zero-shot, fully fine-tuned, and fine-tuned with LoRA, each with 5%, 25% and 100% of the training data.

## Data

[SDWPF](https://arxiv.org/abs/2208.04360) (Baidu KDD Cup 2022): 134 turbines, 10-minute readings, 245 days. Download from Baidu AI Studio or the Hugging Face mirror `aigrids/WindFarm_raw`. Raw data is not committed to this repo.

## Zaratan setup

Access to UMD's Zaratan cluster comes through the course allocation.

1. Log in (MFA is required unless you are on the UMD VPN):
   ```bash
   ssh <umd-directory-id>@login.zaratan.umd.edu
   ```
2. Check the allocation (`msml612-class`) and work in the course scratch space, not home (home is limited to 10 GB):
   ```bash
   sbalance
   cd ~/scratch.msml612
   git clone https://github.com/Sim2200/wind-power-forecasting-chronos.git
   cd wind-power-forecasting-chronos
   ```
3. Create the environment (once, on the login node). It goes in `~/envs/dl612env` (about 5 GB of the 10 GB home quota) because scratch is very slow for the many small files of a PyTorch install; data and results stay in scratch:
   ```bash
   bash scripts/setup_env.sh
   ```
4. Run the GPU smoke test on an A100 MIG slice (7 SU/hour) and check the log:
   ```bash
   sbatch slurm/smoke_test.sbatch
   squeue -u $USER
   cat slurm-<jobid>.out   # expect "available True" and an A100 device name
   ```

GPU cost: 1 A100 hour = 48 SU, 1 MIG slice hour = 7 SU, 1 H100 hour = 144 SU.

## References

- Nie et al. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023.
- Ansari et al. *Chronos: Learning the Language of Time Series.* TMLR 2024.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Zhou et al. *SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array.* Scientific Data 2024.

## Team

- Simran Kharbanda
- Tanmay Sharma
