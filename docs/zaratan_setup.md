# Zaratan setup

Team instructions for running this project on UMD's Zaratan HPC cluster. Access comes through the DATA612 course allocation.

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
