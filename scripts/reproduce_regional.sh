#!/usr/bin/env bash
# Regional fine-tuning: for every variable, split and region, train
# a regional model from scratch and fine-tune the global one, then compare
# global / fine-tuned / local models (and the baseline) on the regional test set.
#   DATA_DIR=/path/to/data scripts/reproduce_regional.sh global|regional|evaluate [--dry_run]
#   global:   train the four global models (CHLA/NITRATE x profile/float)
#   regional: train local and fine-tuned models (needs the global ones in $RUNS)
#   evaluate: compare the models (best.pt of the runs in $RUNS)
# Training jobs are submitted with sbatch (set SBATCH_ARGS, e.g. "--account=MY_ACCOUNT").
# Bay of Bengal has CHLA profile-wise data only; the float-wise NITRATE
# validation set has no North Pacific float.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
DATA_DIR=${DATA_DIR:?set DATA_DIR}
RUNS=${RUNS:-results}
OUT=${OUT:-results/regional_experiments}
STAGE=${1:?usage: $0 global|regional|evaluate [--dry_run]}
DRY_RUN=0; [[ ${2:-} == --dry_run ]] && DRY_RUN=1
REGIONS=(mediterranean north_atlantic north_pacific southern_ocean indian_ocean pacific bay_of_bengal)
declare -A DS=([CHLA]=SEANOE [NITRATE]=CORIOLIS)
declare -A EPOCHS=([CHLA]=200 [NITRATE]=50)          # global and local
declare -A FT_EPOCHS=([CHLA]=100 [NITRATE]=25)       # fine-tuning
declare -A BASELINE=([CHLA]=soca [NITRATE]=canyonb)

submit() {
    if [[ $DRY_RUN -eq 1 ]]; then echo "$*"; else sbatch ${SBATCH_ARGS:-} scripts/slurm_job.sh "$@"; fi
}
run() {
    if [[ $DRY_RUN -eq 1 ]]; then echo "$*"; else "$@"; fi
}

for variable in CHLA NITRATE; do
    ds=${DS[$variable]}
    d=$DATA_DIR/${variable}_datasets
    for split in profile float; do
        train_csv=$d/train_${split}_wise_$ds.csv
        val_csv=$d/val_${split}_wise_$ds.csv
        test_csv=$d/test_${split}_wise_$ds.csv
        global=$RUNS/$variable/global_$split/checkpoints/best.pt
        if [[ $STAGE == global ]]; then
            submit python train.py --variable "$variable" --epochs "${EPOCHS[$variable]}" \
                --train_csv "$train_csv" --val_csv "$val_csv" --output_dir "$RUNS" --run_name "global_$split"
            continue
        fi
        for region in "${REGIONS[@]}"; do
            [[ $region == bay_of_bengal && ( $variable != CHLA || $split != profile ) ]] && continue
            [[ $region == north_pacific && $variable == NITRATE && $split == float ]] && continue
            local_run=local_${region}_$split
            ft_run=ft_${region}_$split
            if [[ $STAGE == regional ]]; then
                submit python train.py --variable "$variable" --region "$region" --epochs "${EPOCHS[$variable]}" \
                    --train_csv "$train_csv" --val_csv "$val_csv" --output_dir "$RUNS" --run_name "$local_run"
                submit python train.py --variable "$variable" --region "$region" --epochs "${FT_EPOCHS[$variable]}" \
                    --init_checkpoint "$global" --ft_strategy aggressive \
                    --train_csv "$train_csv" --val_csv "$val_csv" --output_dir "$RUNS" --run_name "$ft_run"
            else
                baseline=(--baseline "${BASELINE[$variable]}")
                [[ $variable == CHLA ]] && baseline+=(--soca_csv "$d/SOCA_test_$split.csv")
                run python evaluate_regional.py --region "$region" --test_csv "$test_csv" \
                    --model "global=$global" \
                    --model "global_fine_tuned=$RUNS/$variable/$ft_run/checkpoints/best.pt" \
                    --model "local=$RUNS/$variable/$local_run/checkpoints/best.pt" \
                    "${baseline[@]}" --output_dir "$OUT/${variable}_${split}_$region"
                run python evaluate_uncertainty.py --region "$region" --test_csv "$test_csv" --batch_size 1 \
                    --model "global=$global" \
                    --model "global_fine_tuned=$RUNS/$variable/$ft_run/checkpoints/best.pt" \
                    --output_dir "$OUT/uncertainty_${variable}_${split}_$region"
            fi
        done
    done
done
