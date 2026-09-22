# GLOBIO

Train convolutional models for biogeochemical profile prediction and compare global, regional fine-tuned, and regional models for chlorophyll and nitrate.

## Setup

From the repository root, create a virtual environment using Python 3.11 and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the commands below from the repository root with this environment activated. In a new terminal, activate it again with `source .venv/bin/activate`.

The requirements preserve versions from the existing environment, including PyTorch 2.6.0 with CUDA 12.4. Training requires an NVIDIA GPU and a PyTorch installation supporting CUDA and NCCL. Evaluation automatically uses CUDA when available and otherwise runs on CPU.

Datasets and pretrained weights must be provided separately; they are not included in Git. Keep `normalization_stats_CORIOLIS.json` and `normalization_stats_SEANOE.json` in the repository root. Use normalization statistics matching the data and checkpoint being evaluated.

## Dataset paths

Training currently reads from this hard-coded directory in `train_ddp.py`:

```text
/leonardo_work/IscrC_SISCoDEM/globio/old/
├── CHLA_datasets/
│   ├── train_profile_wise_SEANOE.csv
│   └── test_profile_wise_SEANOE.csv
└── NITRATE_datasets/
    ├── train_profile_wise_CORIOLIS.csv
    └── test_profile_wise_CORIOLIS.csv
```

These are example combinations. Filenames are constructed as `train_{split}_wise_{ds_name}.csv` and `test_{split}_wise_{ds_name}.csv`, where `split` is `profile` or `float`. CSVs must follow the prepared profile format expected by `FloatDataset`; arbitrary raw CSVs are not supported.

To train with a different data location, edit `path_ds_train` and `path_ds_val` near the end of `train_ddp.py`. Placing files in `ds/` does not automatically redirect training there.

Evaluation accepts `--data_dir` and `--ds_time`, and reads:

```text
<data_dir>/<ds_time>/CHLA_datasets/test_<split_type>_wise_<ds_name>.csv
<data_dir>/<ds_time>/NITRATE_datasets/test_<split_type>_wise_<ds_name>.csv
```

## Training

Launch with plain `python`: the script spawns one process per visible GPU internally. Do not wrap it in `torchrun` or launch one copy per GPU. On a cluster, run inside a GPU allocation on a single node.

### Global model

Omit `--region` to train on all profiles in the selected CSV:

```bash
python train_ddp.py \
  --variable CHLA \
  --ds_name SEANOE \
  --split profile \
  --total_epochs 200 \
  --save_every 50 \
  --batch_size 26 \
  --lr 0.0001 \
  --dp_rate 0.2 \
  --alpha_smooth_reg 0.0001
```

For nitrate, use `--variable NITRATE --ds_name CORIOLIS`, with the corresponding CSVs available. Training also accepts `BBP700`, but the evaluation scripts documented here cover CHLA and NITRATE.

### Regional model from scratch

Use the same command with a region, for example:

```bash
python train_ddp.py \
  --variable CHLA \
  --ds_name SEANOE \
  --split profile \
  --region mediterranean \
  --total_epochs 200 \
  --save_every 50 \
  --batch_size 26 \
  --lr 0.0001
```

### Regional fine-tuning

Provide both `--checkpoint_path` and `--ft_strategy`. Replace the example checkpoint path with an existing unified checkpoint:

```bash
python train_ddp.py \
  --variable CHLA \
  --ds_name SEANOE \
  --split profile \
  --region mediterranean \
  --checkpoint_path pretrained_models/CHLA/global.pt \
  --ft_strategy aggressive \
  --total_epochs 100 \
  --save_every 25 \
  --batch_size 26 \
  --lr 0.00001
```

- `aggressive`: update all model parameters.
- `freeze_backbone`: freeze the metadata MLPs and selected early CNN layers, with additional BatchNorm freezing logic.

Fine-tuning loads model weights and creates a new optimizer; it does not resume the saved optimizer state or epoch counter.

### Training options and outputs

Always supply `--total_epochs`, `--save_every`, `--split`, and `--ds_name`; their parser defaults are not usable for a complete run. `--batch_size` is per GPU. To restrict visible GPUs, prefix the command with, for example, `CUDA_VISIBLE_DEVICES=0`.

Supported regions are `mediterranean`, `indian_ocean`, `bay_of_bengal`, `southern_ocean`, `north_atlantic`, and `north_pacific`.

Outputs are written under `results_trial_ddp/<VARIABLE>/<run-directory>/`, including loss history, run information, periodic checkpoints in `model/`, and best-model checkpoints in `model_best/`.

For evaluation and fine-tuning, use a unified `checkpoint_<epoch>.pt` containing `model`, `mlp_day`, `mlp_year`, and `mlp_space`. The separate files such as `model_best.pt` contain individual state dictionaries and cannot be passed directly to these commands.

## Evaluation

Both scripts require three unified checkpoints: global, regional fine-tuned, and regional trained from scratch. The paths below are placeholders; replace them with your actual files. Use checkpoints matching the variable, normalization, and intended split.

### Chlorophyll

```bash
python evaluate_chla.py \
  --path_global_model pretrained_models/CHLA/global.pt \
  --path_ft_model pretrained_models/CHLA/finetuned_mediterranean.pt \
  --path_local_model pretrained_models/CHLA/local_mediterranean.pt \
  --main_dir . \
  --data_dir /leonardo_work/IscrC_SISCoDEM/globio \
  --ds_time old \
  --ds_name SEANOE \
  --split_type profile \
  --region mediterranean \
  --exp_name chla_profile_mediterranean
```

### Nitrate

```bash
python evaluate_nitrate.py \
  --path_global_model pretrained_models/NITRATE/global.pt \
  --path_ft_model pretrained_models/NITRATE/finetuned_mediterranean.pt \
  --path_local_model pretrained_models/NITRATE/local_mediterranean.pt \
  --main_dir . \
  --data_dir /leonardo_work/IscrC_SISCoDEM/globio \
  --ds_time old \
  --ds_name CORIOLIS \
  --split_type profile \
  --region mediterranean \
  --exp_name nitrate_profile_mediterranean
```

Evaluation prints per-model RMSE summaries, a ranking by median RMSE, and Wilcoxon comparisons with Holm correction. Profile plots are saved under `<main_dir>/experiments/<exp_name>/`. Nitrate metric evaluation currently uses a limit of 700 samples in the script.

### Optional baselines

Add `--compute_baseline True` to either evaluation command:

- CHLA compares against SOCA and requires `<data_dir>/<ds_time>/CHLA_datasets/SOCA_test_<split_type>_old.csv`. The SOCA profiles must align with the evaluated profiles; supply a supported `--region`.
- NITRATE computes the CANYON-B baseline using `canyonbpy`.

To disable baselines, **omit the argument**. The current parser uses `type=bool`, so `--compute_baseline False` also enables it.

`canyonbpy` and Cartopy are imported by the evaluation modules even when baselines or maps are not requested. They remain required with the current code. Map utilities also contain a local Cartopy data path in `evaluation/plot.py`; adjust it if using those utilities on another machine.

## Command-line help

```bash
python train_ddp.py --help
python evaluate_chla.py --help
python evaluate_nitrate.py --help
```

These commands still require the imported dependencies to be installed. The examples document the current CLI; they have not been validated by a complete training or evaluation run.
