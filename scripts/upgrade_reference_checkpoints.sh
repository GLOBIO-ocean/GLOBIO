#!/usr/bin/env bash
# Convert the reference checkpoints (weights only) into self-contained
# checkpoints with variable and normalization statistics.
#   scripts/upgrade_reference_checkpoints.sh SOURCE_DIR [DEST_DIR]
# SOURCE_DIR contains CHLA/*.pt and NITRATE/*.pt (e.g. global_profile_checkpoint_116.pt);
# DEST_DIR defaults to pretrained_models. CHLA models were trained on SEANOE
# (statistics SEANOE_CHLA) except those with "coriolis" in the name (CORIOLIS_CHLA);
# NITRATE models on CORIOLIS (CORIOLIS_NITRATE).
set -euo pipefail
src=${1:?usage: $0 SOURCE_DIR [DEST_DIR]}
dest=${2:-pretrained_models}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
for variable in CHLA NITRATE; do
    for checkpoint in "$src/$variable"/*.pt; do
        [[ -e $checkpoint ]] || continue
        name=$(basename "$checkpoint")
        if [[ $variable == NITRATE ]]; then stats=CORIOLIS_NITRATE
        elif [[ $name == *coriolis* ]]; then stats=CORIOLIS_CHLA
        else stats=SEANOE_CHLA; fi
        out="$dest/$variable/$name"
        [[ -e $out ]] && { echo "exists: $out"; continue; }
        python -m tools.upgrade_checkpoint --checkpoint "$checkpoint" --variable "$variable" \
            --stats_path "normalization_stats/$stats.json" --output "$out" --note "statistics: $stats"
    done
done
