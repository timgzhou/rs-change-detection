#!/bin/bash
# Fan out one sbatch job per cached feature set, each running all LP head types via
# lp_heads.sh. Run from a login node (NOT under sbatch):
#   ./submit_lp_heads.sh
# Override the feature list by passing them as arguments:
#   ./submit_lp_heads.sh oe_base_s2_ps1_tile1 oe_base_s2s1_ps4_tile64
#
# Gate every LP job on an extraction job finishing successfully (afterok). Pass the
# extraction job id via DEPEND, e.g.:
#   DEPEND=4011001 ./submit_lp_heads.sh oe_base_s2_ps1_tile1
# To gate each feature set on its OWN extraction job, run this once per feature set with
# the matching DEPEND (or just submit the LP job by hand with --dependency).

FEATURES=("$@")
if [ ${#FEATURES[@]} -eq 0 ]; then
    FEATURES=(
        # oe_base_s2_ps1_tile1
        # oe_base_s2_ps1_tile8
        # oe_base_s2_ps2_tile32
        # oe_base_s2_ps4_tile64
        # oe_base_s2s1_ps4_tile64
        oe_base_s2_ps2_tile8
        oe_base_s2_ps4_tile8
    )
fi

# Optional dependency: only start LP jobs after extraction job(s) DEPEND finish OK.
DEP_ARG=()
if [ -n "$DEPEND" ]; then
    DEP_ARG=(--dependency="afterok:${DEPEND}")
    echo "gating all LP jobs on afterok:${DEPEND}"
fi

for feat in "${FEATURES[@]}"; do
    echo "submitting lp_heads.sh for $feat"
    sbatch "${DEP_ARG[@]}" --export=ALL,FEATURES="$feat" lp_heads.sh
done

# bash submit_lp_heads.sh