#!/usr/bin/env bash
# Global models (profile-wise split): training and evaluation.
#   DATA_DIR=/path/to/data MODELS_DIR=pretrained_models scripts/reproduce_global.sh train|evaluate
# DATA_DIR contains CHLA_datasets/ and NITRATE_datasets/ with the CSVs
# {train,val,test}_profile_wise_{SEANOE,CORIOLIS}.csv and, for the SOCA
# baseline, CHLA_datasets/SOCA_test_profile.csv (tools.download_soca).
# MODELS_DIR holds the global checkpoints (self-contained, see
# scripts/upgrade_reference_checkpoints.sh). Training is submitted with sbatch
# (set SBATCH_ARGS, e.g. "--account=MY_ACCOUNT"); evaluation runs here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_DIR=${DATA_DIR:?set DATA_DIR}
MODELS_DIR=${MODELS_DIR:-pretrained_models}
OUT=${OUT:-results/global_experiments}
C=$DATA_DIR/CHLA_datasets
N=$DATA_DIR/NITRATE_datasets
# variable  dataset  epochs  checkpoint
CONFIGS=(
    "CHLA SEANOE 200 $MODELS_DIR/CHLA/global_profile_checkpoint_116.pt"
    "CHLA CORIOLIS 200 $MODELS_DIR/CHLA/global_profile_coriolis_checkpoint_181.pt"
    "NITRATE CORIOLIS 50 $MODELS_DIR/NITRATE/global_profile_checkpoint_36.pt"
)
case ${1:-} in
train)
    for config in "${CONFIGS[@]}"; do
        read -r variable ds epochs _ <<<"$config"
        d=$DATA_DIR/${variable}_datasets
        sbatch ${SBATCH_ARGS:-} --job-name="global_${variable}_${ds}" scripts/slurm_job.sh \
            python train.py --variable "$variable" --epochs "$epochs" \
            --train_csv "$d/train_profile_wise_$ds.csv" --val_csv "$d/val_profile_wise_$ds.csv" \
            --output_dir results --run_name "global_profile_$ds"
    done ;;
evaluate)
    for config in "${CONFIGS[@]}"; do
        read -r variable ds _ checkpoint <<<"$config"
        d=$DATA_DIR/${variable}_datasets
        python evaluate_global.py --checkpoint "$checkpoint" --depth_stats \
            --data "train=$d/train_profile_wise_$ds.csv" --data "val=$d/val_profile_wise_$ds.csv" \
            --data "test=$d/test_profile_wise_$ds.csv" --output_dir "$OUT/global_${variable}_${ds}"
        python evaluate_uncertainty.py --model "global=$checkpoint" --test_csv "$d/test_profile_wise_$ds.csv" \
            --batch_size 1 --output_dir "$OUT/uncertainty_${variable}_${ds}"
    done
    # Baselines on the test sets
    python evaluate_global.py --checkpoint "$MODELS_DIR/CHLA/global_profile_checkpoint_116.pt" \
        --data "test=$C/test_profile_wise_SEANOE.csv" --baseline soca --soca_csv "$C/SOCA_test_profile.csv" \
        --output_dir "$OUT/global_CHLA_SEANOE_soca"
    python evaluate_global.py --checkpoint "$MODELS_DIR/NITRATE/global_profile_checkpoint_36.pt" \
        --data "test=$N/test_profile_wise_CORIOLIS.csv" --baseline canyonb \
        --output_dir "$OUT/global_NITRATE_CORIOLIS_canyonb" ;;
*)
    echo "usage: DATA_DIR=... $0 train|evaluate" >&2; exit 1 ;;
esac
