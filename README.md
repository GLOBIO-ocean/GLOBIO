# GlobIO

GlobIO predicts vertical profiles of biogeochemical variables (chlorophyll
`CHLA`, nitrate `NITRATE`, particle backscattering `BBP700`) from the physical
profiles measured by BGC-Argo floats (temperature, salinity, dissolved
oxygen), the position and the date, with one-dimensional convolutional
networks. A global model is trained on the whole ocean; it can be fine-tuned
on a region and compared with regional models trained from scratch. Model
uncertainty is estimated with Monte Carlo (MC) dropout.

This repository contains everything used in this work:

* Training of the **global models** and their
  evaluation by season, by latitude band and by depth, against the SOCA (CHLA)
  and CANYON-B (NITRATE) baselines, with MC dropout uncertainty.
* **Fine-tuning** of the global models on seven
  regions, comparison with regional models trained from scratch and with the
  baselines, and statistical tests.

Contents:

1. [Installation](#installation)
2. [Repository layout](#repository-layout)
3. [Data](#data)
4. [Pretrained models](#pretrained-models)
5. [Training](#training)
6. [Inference](#inference)
7. [Uncertainty with MC dropout](#uncertainty-with-mc-dropout)
8. [Evaluation](#evaluation)
9. [Reproducing the experiments](#reproducing-the-experiments)
10. [Tests](#tests)
11. [Notes on the original experiment code](#notes-on-the-original-experiment-code)

## Installation

Python 3.9 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt   # versions used for the reference runs (PyTorch 2.6, CUDA 12.4)
python -m pip install -e .                  # makes `import globio` work from any directory
```

Only `torch`, `numpy`, `pandas` and `scipy` are required. The other packages
are optional: `canyonbpy` for the CANYON-B baseline, `scikit-learn` for
`tools.split_dataset`, `netCDF4` and `copernicusmarine` for
`tools.make_dataset` and `tools.download_soca`, `matplotlib` for the
notebook. Instead of `requirements.txt` you can install the package with the
extras you need, e.g. `pip install -e ".[baselines,notebook]"` (pick the
PyTorch build for your CUDA version from <https://pytorch.org>).

Training uses all the visible GPUs of a node; it also runs (slowly) on CPU.
Inference and evaluation use a GPU when available and the CPU otherwise.

## Repository layout

| Path | Content |
|---|---|
| `train.py` | training, regional training and fine-tuning |
| `predict.py` | predictions for a dataset CSV, optionally with MC dropout |
| `evaluate_global.py` | evaluation of a global model |
| `evaluate_regional.py` | comparison of global, fine-tuned and regional models |
| `evaluate_uncertainty.py` | calibration of the MC dropout uncertainty |
| `globio/` | the library: data (`data.py`), networks and checkpoints (`models.py`), training (`training.py`), inference and MC dropout (`inference.py`), metrics and tests (`metrics.py`), baselines (`baselines.py`), constants and regions (`constants.py`) |
| `tools/` | dataset creation, cleaning, splitting, SOCA download, checkpoint conversion |
| `normalization_stats/` | input normalization statistics of the reference models |
| `notebooks/inference_example.ipynb` | inference, MC dropout and profile plots |
| `scripts/` | SLURM job template and the commands to reproduce the experiments |
| `tests/` | unit and end-to-end tests on synthetic data |

Every script has a `--help`. Tools run from the repository root as
`python -m tools.<name>`.

## Data

### Dataset CSV format

A dataset is a CSV file with **one column per profile** and eight rows:

```text
,o_scratch/.../5906529/SD5906529_081,o_scratch/.../2903468/SD2903468_010,...
year,2023,2023,...
day_rad,1.5837,4.7855,...
lat,21.7971,-18.6451,...
lon,-157.0802,99.7557,...
temp,"tensor([24.1590, 24.1590, ..., 9.1350])",...
psal,"tensor([34.8020, ..., 34.4210])",...
doxy,"tensor([210.8183, ..., 87.0412])",...
CHLA,"tensor([0.0253, ..., 0.0012])",...
```

* The header holds the profile ids, paths ending in `<float id>/<profile name>`;
  the float id (parent directory) is used by the float-wise splits.
* `year` is the calendar year; `day_rad = ((month - 1) * 30 + day) * 2 pi / 365`
  encodes the day with 30-day pseudo-months (use
  `globio.day_rad_from_date(date)`); `lat` and `lon` are in degrees
  (longitude in [-180, 180]).
* Each profile has **200 levels** on a fixed pressure grid that depends on the
  target: 0-199 dbar every 1 dbar for CHLA and BBP700, 0-995 dbar every 5 dbar
  for NITRATE (`globio.pressure_levels(variable)`). Values are comma-separated,
  optionally inside `tensor([...])` or `[...]`.
* The last row is the target, labelled with the variable name. It can be
  omitted to predict new profiles. The first column (row labels) is optional.
* Units: temperature °C, practical salinity, oxygen µmol kg⁻¹, CHLA mg m⁻³,
  NITRATE mmol m⁻³, BBP700 10⁻³ m⁻¹.

When a CSV is loaded (`globio.ProfileDataset`), profiles with inputs outside
the Argo global range (temperature −2.5…40, salinity 2…41, oxygen −5…600) or
with non-finite or fill values (≥ 99999) in inputs or target are dropped.

The model sees `year - 2010`, `(sin, cos)` of `day_rad`, the position on the
unit sphere `(cos lat cos lon, cos lat sin lon, sin lat)` and the normalized
temperature, salinity and oxygen profiles.

### Normalization statistics

Inputs are standardized with the mean and standard deviation of temperature,
salinity and oxygen over the training set (all levels of all profiles), stored
as JSON:

```json
{"temp": [12.07, 8.78], "psal": [34.84, 2.36], "doxy": [234.26, 74.91]}
```

`train.py` computes them on `--train_csv` (before any region selection, so
global, regional and fine-tuned models of a dataset share them) and stores
them in the run directory and in every checkpoint. Fine-tuning reuses the
statistics of the pretrained model. A model must always be used with its own
statistics: the scripts take them from the checkpoint.

### Building the datasets

The datasets of this work were built from the quality-controlled BGC-Argo
collections SEANOE (CHLA) and CORIOLIS (NITRATE, and CHLA for the CORIOLIS CHLA
model) with the steps below, in this order (`tools.make_dataset` shuffles
with seed 123, `tools.split_dataset` with seed 42, their defaults);
`tools/make_dataset.py` reproduces the CSV values of the reference datasets. Example for
CHLA SEANOE (`D=data/CHLA_datasets`):

```bash
# 1. NetCDF profiles -> CSV with a random 80/20 profile-wise train/test split
python -m tools.make_dataset --input_dir /path/to/SEANOE_QCed --variable CHLA --output_prefix $D/SEANOE
mv $D/SEANOE_train.csv $D/train_profile_wise_SEANOE.csv
mv $D/SEANOE_test.csv $D/test_profile_wise_SEANOE.csv   # SEANOE_removed.csv: profiles rejected by the QC

# 2. Float-wise split of the same profiles: 20% of the floats to test
python -m tools.split_dataset $D/train_profile_wise_SEANOE.csv $D/test_profile_wise_SEANOE.csv \
    --split float --fraction 0.2 --out_rest $D/train_float_wise_SEANOE.csv --out_fraction $D/test_float_wise_SEANOE.csv

# 3. Remove corrupted profiles (broken sensors, fill values, constant profiles); keeps .bak copies
python -m tools.clean_dataset $D/{train,test}_{profile,float}_wise_SEANOE.csv --inplace
#    (NITRATE used a wider target range: --target_range -10 60)

# 4. Validation sets: half of each test set (80/10/10 overall)
for split in profile float; do
    mv $D/test_${split}_wise_SEANOE.csv $D/test_${split}_wise_SEANOE_full.csv
    python -m tools.split_dataset $D/test_${split}_wise_SEANOE_full.csv --split $split --fraction 0.5 \
        --out_rest $D/test_${split}_wise_SEANOE.csv --out_fraction $D/val_${split}_wise_SEANOE.csv
done
```

The split tools never overwrite files. `tools.select_profiles` selects the
profiles of some years and/or splits them by the floats of a reference
training set (used for the evaluation on the 2025 profiles).

The validation set is used only to choose the best epoch; report results on
the test set.

## Pretrained models

The reference checkpoints are distributed separately. They contain only
the network weights, so they must be used with the matching normalization
statistics of `normalization_stats/`:

| Models | Training data | Statistics |
|---|---|---|
| CHLA global, fine-tuned and regional | SEANOE | `normalization_stats/SEANOE_CHLA.json` |
| CHLA global (`*coriolis*`) | CORIOLIS | `normalization_stats/CORIOLIS_CHLA.json` |
| NITRATE global, fine-tuned and regional | CORIOLIS | `normalization_stats/CORIOLIS_NITRATE.json` |

Convert them once into self-contained checkpoints (variable, dropout rate and
statistics stored inside), so that `--variable` and `--stats_path` are no
longer needed:

```bash
scripts/upgrade_reference_checkpoints.sh /path/to/model_pretrained pretrained_models
# or one at a time:
python -m tools.upgrade_checkpoint --checkpoint global_profile_checkpoint_116.pt --variable CHLA \
    --stats_path normalization_stats/SEANOE_CHLA.json --output pretrained_models/CHLA/global_profile_checkpoint_116.pt
```

Old checkpoints can also be used directly by passing `--variable` (needed for
CHLA/BBP700; NITRATE is recognized from the weights) and `--stats_path` to
any script.

## Training

`train.py` runs one process per visible GPU (DistributedDataParallel, NCCL);
the batch size is per GPU. Launch it with plain `python` on one node (restrict
the GPUs with `CUDA_VISIBLE_DEVICES`); for several nodes use
`torchrun --nnodes N --nproc_per_node G train.py ...`. Without GPUs it runs on
CPU (`--device cpu`, `--n_procs` processes).

### Global model

```bash
python train.py --variable CHLA \
    --train_csv data/CHLA_datasets/train_profile_wise_SEANOE.csv \
    --val_csv data/CHLA_datasets/val_profile_wise_SEANOE.csv \
    --epochs 200 --output_dir results --run_name global_profile
```

### Regional model from scratch

Add `--region`: training and validation use only the profiles of the region.

```bash
python train.py --variable CHLA --region mediterranean --epochs 200 \
    --train_csv data/CHLA_datasets/train_profile_wise_SEANOE.csv \
    --val_csv data/CHLA_datasets/val_profile_wise_SEANOE.csv --run_name local_mediterranean_profile
```

Regions (inclusive boxes, `globio/constants.py`): `mediterranean`
(30–46°N, 6°W–36°E), `north_atlantic` (45–70°N, 70°W–10°E), `north_pacific`
(45–65°N, 180–120°W), `southern_ocean` (south of 60°S), `indian_ocean`
(40°S–30°N, 20–120°E), `bay_of_bengal` (8–22°N, 82–98°E), `pacific`
(60°S–65°N, 120°E–70°W), and the latitude bands `r1`…`r6` (60–90°N, 30–60°N,
0–30°N, 0–30°S, 30–60°S, 60–90°S).

### Fine-tuning

Start from a trained model with `--init_checkpoint`:

```bash
python train.py --variable CHLA --region mediterranean --epochs 100 \
    --init_checkpoint results/CHLA/global_profile/checkpoints/best.pt \
    --train_csv data/CHLA_datasets/train_profile_wise_SEANOE.csv \
    --val_csv data/CHLA_datasets/val_profile_wise_SEANOE.csv --run_name ft_mediterranean_profile
```

`--ft_strategy aggressive` (default, used in the reference runs) trains all the
weights; `freeze_backbone` freezes the metadata MLPs, the first two CNN blocks
and all BatchNorm layers. The optimizer starts afresh. To fine-tune an old
checkpoint, add `--stats_path` with its statistics.

### Options and reference settings

| Option | Default | Meaning |
|---|---|---|
| `--epochs` | 200 | number of epochs |
| `--batch_size` | 32 | batch size per GPU (reference runs: 4 GPUs) |
| `--lr` | 1e-3 | Adam learning rate |
| `--dp_rate` | 0.2 | dropout rate (also used by MC dropout) |
| `--alpha_smooth_reg` | 1.99e-2 | weight of the smoothness penalty |
| `--seed` | 0 | seed of initialization and shuffling |
| `--save_every` | 0 | also keep a checkpoint every N epochs |
| `--recalibrate_bn` | off | recompute BatchNorm statistics on the training set before each validation |
| `--stats_path` | — | normalization statistics (default: computed, or those of `--init_checkpoint`) |
| `--max_profiles` | — | use only the first profiles (quick tests) |

The loss is

```text
MSE(prediction, target) + alpha_smooth_reg * sum_profiles [ 1/199 * sum_i |p[i+1] - p[i]| ]
```

i.e. the mean squared error over the batch plus, for each profile, the mean
absolute difference between consecutive levels, summed over the profiles of
the batch. The reference value is `alpha_smooth_reg = 1.99e-2`.

Epochs of the reference runs (all other options at their defaults, aggressive
fine-tuning):

| | global | regional from scratch | fine-tuning |
|---|---:|---:|---:|
| CHLA | 200 | 200 | 100 |
| NITRATE | 50 | 50 | 25 |

### Outputs

A run is saved in `<output_dir>/<VARIABLE>/<run_name>/` (default run name:
date, time and kind of run):

```text
config.json                 options, git commit, PyTorch version
normalization_stats.json    input statistics
history.json                training and validation loss per epoch, best epoch
checkpoints/best.pt         model with the lowest validation loss (use this one)
checkpoints/last.pt         last epoch, with optimizer state (to resume)
checkpoints/epoch_NNNN.pt   with --save_every N
```

An interrupted run continues with `python train.py --resume <run_dir>`
(optionally with a larger `--epochs`).

On a SLURM cluster submit, for example,

```bash
sbatch --account=MY_ACCOUNT scripts/slurm_job.sh python train.py --variable CHLA ...
```

(edit the `#SBATCH` defaults in `scripts/slurm_job.sh` for your cluster).

## Inference

### Command line

```bash
python predict.py --checkpoint pretrained_models/CHLA/global_profile_checkpoint_116.pt \
    --csv data/CHLA_datasets/test_profile_wise_SEANOE.csv --output results/predictions/chla_test
```

writes `chla_test.npz` (arrays `profile_id`, `year`, `day_rad`, `lat`, `lon`,
`pressure`, `prediction` of shape `(n_profiles, 200)` and `target` if the CSV
has it), `chla_test.csv` (one row per profile with metadata and, with
targets, the profile-wise RMSE) and `chla_test.json` (settings). Add
`--region` to predict one region, `--mc_samples 100` for MC dropout (below).
The target row is optional, so new profiles can be predicted.

### Python

```python
from globio import ProfileDataset, load_checkpoint, predict, resolve_norm_stats

model, info = load_checkpoint("pretrained_models/CHLA/global_profile_checkpoint_116.pt", device="cuda")
stats = resolve_norm_stats(info)          # or resolve_norm_stats(info, "normalization_stats/SEANOE_CHLA.json")
dataset = ProfileDataset("data/test.csv", model.variable, stats, region="mediterranean")
prediction = predict(model, dataset)      # numpy array (n_profiles, 200), physical units
```

Profiles that are not in a CSV can be given as arrays on the 200-level grid:

```python
import datetime as dt
from globio import ProfileDataset, day_rad_from_date

dataset = ProfileDataset.from_arrays(
    year=[2024], day_rad=[day_rad_from_date(dt.date(2024, 5, 15))], lat=[40.0], lon=[5.0],
    temp=temp, psal=psal, doxy=doxy,            # arrays of shape (n_profiles, 200)
    variable=model.variable, norm_stats=stats)
```

See `notebooks/inference_example.ipynb` for a complete example with plots of
predicted and measured profiles.

## Uncertainty with MC dropout

MC dropout keeps the dropout layers active at inference: each profile is
predicted `N` times with different random dropout masks, while BatchNorm uses
its running statistics. The mean of the `N` predictions is the estimate and
their standard deviation σ (per level, ddof=1) the uncertainty of the model.
The reference experiments use `N = 100` with the training dropout rate (0.2). σ describes
the model (epistemic) uncertainty, not the measurement error.

```bash
python predict.py --checkpoint best.pt --csv test.csv --output results/pred_mc --mc_samples 100
```

adds `mc_mean` and `mc_std` to the NPZ (`--save_samples` also stores every
pass) and the RMS of σ over the levels (`uncertainty_rms`) to the CSV.

```python
from globio import predict_mc

mc = predict_mc(model, dataset, n_samples=100, batch_size=32, seed=123)
mc["mean"], mc["std"], mc["deterministic"]     # each (n_profiles, 200)
```

Results are reproducible for a given seed, batch size and device (CPU and GPU
draw different random numbers).

To assess the calibration on a test set:

```bash
python evaluate_uncertainty.py --model global=pretrained_models/CHLA/global_float_checkpoint_145.pt \
    --test_csv data/CHLA_datasets/test_float_wise_SEANOE.csv --batch_size 1 --output_dir results/unc_chla_float
# global versus fine-tuned on a region
python evaluate_uncertainty.py --region mediterranean --test_csv data/CHLA_datasets/test_profile_wise_SEANOE.csv \
    --model global=global.pt --model global_fine_tuned=ft_mediterranean.pt --output_dir results/unc_med
```

Outputs, per model: `profiles_<name>.csv` (RMSE of the MC mean, RMS of σ,
whether all levels are within k σ and the fraction of levels that are, k = 1,
2, 3), `coverage_<name>.csv` (percentage of profiles entirely covered and
point-wise coverage, i.e. level fractions averaged per profile),
`vertical_<name>.csv` (median |error| and median σ per pressure level), plus
`coverage.csv` with all models and `predictions.npz`. Recompute the tables
from the archive with `--results <dir>/predictions.npz`.

## Evaluation

All errors are in physical units; the **profile-wise RMSE** is the RMSE over
the levels of one profile. Every evaluation script writes per-profile CSVs, so
other statistics and figures can be computed from them.

### Global model

```bash
python evaluate_global.py --checkpoint pretrained_models/CHLA/global_profile_checkpoint_116.pt \
    --data train=data/CHLA_datasets/train_profile_wise_SEANOE.csv \
    --data val=data/CHLA_datasets/val_profile_wise_SEANOE.csv \
    --data test=data/CHLA_datasets/test_profile_wise_SEANOE.csv \
    --depth_stats --output_dir results/global_experiments/global_CHLA_SEANOE
```

For each `--data` set: profile-wise RMSE summarized (mean, median, standard
deviation) over all profiles, summer, winter (seasons inverted south of the
equator) and the latitude bands r1–r6 (`summary.csv`, `profiles_<set>.csv`),
and with `--depth_stats` the depth-resolved errors per band and season (mean
absolute error and its SD, bias, RMSE at each level: `depth_<set>.csv`).

Baselines, on the set named by `--baseline_on` (default `test`):

* `--baseline soca --soca_csv SOCA.csv` (CHLA): SOCA and the model on the 22
  SOCA depths, on the profiles with a valid SOCA estimate.
* `--baseline canyonb` (NITRATE): CANYON-B on all levels, plus the analysis of
  the CANYON-B uncertainty range NO3 ± NO3_ci (how often predictions and
  measurements fall inside it).

Baseline outputs: `baseline_summary.csv`, `baseline_tests.csv` (Wilcoxon test,
below) and for CANYON-B `baseline_canyonb_range_*.csv`.

### Global, fine-tuned and regional models

```bash
python evaluate_regional.py --region mediterranean \
    --test_csv data/CHLA_datasets/test_profile_wise_SEANOE.csv \
    --model global=pretrained_models/CHLA/global_profile_checkpoint_116.pt \
    --model global_fine_tuned=pretrained_models/CHLA/ft_mediterranean_profile_checkpoint_77.pt \
    --model local=pretrained_models/CHLA/local_mediterranean_profile_checkpoint_173.pt \
    --baseline soca --soca_csv data/CHLA_datasets/SOCA_test_profile.csv \
    --output_dir results/regional_experiments/CHLA_profile_mediterranean
```

Any number of `--model NAME=CHECKPOINT` can be compared. Methods are ranked
by median profile-wise RMSE; a paired one-sided Wilcoxon signed-rank test
(errors rounded to 2 decimals, `--wilcoxon_decimals`), Holm-corrected over
the comparisons of each method, says whether the best one is significantly
better than all the others (α = 0.05, `--alpha`). Two comparisons are
written: `comparison_models.csv` (models only, all levels and profiles) and
`comparison_with_baseline.csv` (with SOCA on its depths and valid profiles,
or with CANYON-B on all levels). Per-profile errors are in `profiles.csv`.

### Baselines

* **CANYON-B** (Bittig et al., 2018) is computed with `canyonbpy` from the same
  temperature, salinity, oxygen, position and date.
* **SOCA** CHLA profiles (Copernicus product
  `cmems_obs-mob_glo_bgc-chl-poc_my_0.25deg_P7D-m`) must be downloaded once
  for every profile of a test set (Copernicus Marine account required,
  `copernicusmarine login`):

  ```bash
  python -m tools.download_soca --csv data/CHLA_datasets/test_profile_wise_SEANOE.csv \
      --cache_dir data/soca_cache/profile --output_csv data/CHLA_datasets/SOCA_test_profile.csv
  ```

  The download can be interrupted and restarted. SOCA profiles are matched to
  the test profiles by profile id. Profiles outside the product (e.g. after
  its last date) are excluded from the comparison.

### Variability of the regional training data

`python -m tools.training_variability --csv CHLA_profile=train_profile_wise_SEANOE.csv ... --output_dir out`
reports, for each region, the number of training profiles and the coefficient
of variation of the target, used to relate the gain of fine-tuning
to the regional training data.

## Reproducing the experiments

`scripts/reproduce_global.sh` and `scripts/reproduce_regional.sh` contain the
complete sequence of training and evaluation commands (training jobs are
submitted with `sbatch scripts/slurm_job.sh`). `DATA_DIR` must contain
`CHLA_datasets/` and `NITRATE_datasets/` with the CSVs named
`{train,val,test}_{profile,float}_wise_{SEANOE,CORIOLIS}.csv` and the SOCA
files `SOCA_test_{profile,float}.csv`:

```bash
DATA_DIR=/path/to/data scripts/reproduce_regional.sh global --dry_run    # print the commands
DATA_DIR=/path/to/data SBATCH_ARGS="--account=MY_ACCOUNT" scripts/reproduce_regional.sh global
DATA_DIR=/path/to/data scripts/reproduce_regional.sh regional            # after the global models
DATA_DIR=/path/to/data scripts/reproduce_regional.sh evaluate
DATA_DIR=/path/to/data MODELS_DIR=pretrained_models scripts/reproduce_global.sh evaluate
```

Evaluating the reference checkpoints with this code reproduces the reference
numbers (e.g. Mediterranean CHLA, profile split: median RMSE 0.0391 local,
0.0403 SOCA, 0.0409 fine-tuned, 0.0462 global). Differences of the order of
10⁻⁴ appear between GPU and CPU runs because of the GPU arithmetic (TF32
convolutions). Retraining gives statistically equivalent, not identical,
models.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

The tests use synthetic data and run on CPU in about a minute: data parsing
and selection, encodings, networks and checkpoints, MC dropout, metrics and
statistical tests, training (also with two processes), fine-tuning, resume,
and every script and tool.

## Notes on the original experiment code

The code was reorganized from the original experiment code. Models, data processing, metrics
and baselines are unchanged (the reference checkpoints give the same
predictions), with these differences in training:

* Epochs are numbered from 1 and exactly `--epochs` epochs are run (the old
  script ran `total_epochs + 1`, numbered from 0); the best model is a single
  `best.pt` instead of `model_best/checkpoint_<epoch>.pt`.
* Checkpoints also store the variable, the dropout rate and the normalization
  statistics; old checkpoints remain loadable.
* In fine-tuning with several GPUs the metadata MLPs are now synchronized
  across GPUs like the CNN (the old script left them unsynchronized).
* `freeze_backbone` now keeps the BatchNorm layers frozen during the whole
  fine-tuning (this strategy was not used in the reference runs).
* The smoothness penalty is now divided by the number of level differences
  (199), as in the formula of the method; the old code omitted the division
  and used the weight 1e-4 instead of 1.99e-2. The two losses are numerically
  identical (1.99e-2 / 199 = 1e-4).
* The validation loss is computed on the whole validation set by one process,
  so it does not depend on the number of GPUs (the old script averaged over
  per-GPU shards padded with repeated profiles).
* The validation set is given explicitly with `--val_csv`, and the paths of
  data and outputs are command-line options.
