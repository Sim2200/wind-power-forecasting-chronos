# Wind Power Forecasting: Do Time-Series Foundation Models Need Fine-Tuning?

**DATA612 PWS1 Group Project**: Deep Learning (DATA/MSML 612), Fall 2026, University of Maryland.

We forecast wind-farm power with a PatchTST transformer implemented from scratch, and compare it against the pretrained Chronos foundation model used zero-shot, fully fine-tuned, and fine-tuned with LoRA, each with 5%, 25% and 100% of the training data.

## Data

[SDWPF](https://arxiv.org/abs/2208.04360) (Baidu KDD Cup 2022): 134 turbines, 10-minute readings, 245 days. Download from Baidu AI Studio or the Hugging Face mirror `aigrids/WindFarm_raw`. Raw data is not committed to this repo.

## References

- Nie et al. *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023.
- Ansari et al. *Chronos: Learning the Language of Time Series.* TMLR 2024.
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Zhou et al. *SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting over a Large Turbine Array.* Scientific Data 2024.

## Team

- Simran Kharbanda
- Tanmay Sharma
