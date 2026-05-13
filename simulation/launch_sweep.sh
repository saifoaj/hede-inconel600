#!/bin/bash
# ============================================================================
# Fire all 7 sweep jobs in parallel. Each `sbatch` enqueues an independent
# job — they will run concurrently as queue slots open instead of one-after-
# another inside a single allocation.
#
# Why parallel beats sequential here: total wallclock collapses from
# 28-84 h (sequential) to roughly the slowest single job (~12 h) plus queue
# wait. License token cost is trivial: 7 jobs * 9 tokens = 63 / 450 cap.
# Resilience also improves -- one job crashing does not block the rest.
#
# Usage:  bash launch_sweep.sh
# Then:   squeue -u $USER          # see all 7 jobs
#         squeue --start -u $USER  # estimated start times
# ============================================================================

JOBS=(c0_0 c0_0p5 c0_1p0 c0_1p5 c0_2p0 c0_3p0 c0_5p0)

echo "Submitting ${#JOBS[@]} parallel sweep jobs to multicore partition..."
echo

for J in "${JOBS[@]}"; do
    sbatch \
        -J hede_$J \
        -o hede_${J}_%j.out \
        -e hede_${J}_%j.err \
        submit_one.sh $J
done

echo
echo "All jobs submitted. Check status with:"
echo "    squeue -u \$USER"
echo "    squeue --start -u \$USER       # estimated start times per job"
echo
echo "Outputs land in ./jobs/ (per-job .odb .sta .msg .dat)."
echo "Slurm stdout/stderr land in current dir as hede_<deck>_<jobid>.{out,err}."
