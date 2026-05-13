#!/bin/bash --login
# ============================================================================
# Single-job runner for one C0 wppm deck. Called by launch_sweep.sh, not
# directly. The deck name (e.g. c0_0p5) is passed as the first positional
# argument: `sbatch submit_one.sh c0_0p5`.
#
# Run by `bash launch_sweep.sh` which fires 7 parallel sbatch invocations
# (one per deck) so the entire sweep runs concurrently in the queue rather
# than back-to-back in a single allocation.
# ============================================================================

#SBATCH -p multicore        # AMD Genoa nodes; 8 GB/core; 2-168 cores allowed
#SBATCH -n 4                # 4 cores. Abaqus/Standard's direct sparse solver
                            # uses thread parallelism and scales 2-3x going
                            # from 1 to 4 threads on a 396k-DOF 2D problem.
                            # USDFLD also parallelises across elements when
                            # extra threads are available. 8 cores would add
                            # only ~20% more for higher token cost and worse
                            # queue contention -- 4 is the sweet spot here.
#SBATCH -t 1-00:00:00       # 24 h. Per-job worst case is ~12 h; 2x safety
                            # means a slow-converging case finishes instead
                            # of getting killed mid-run.
# Note: -J (jobname) and -o/-e (output paths) are set on the sbatch
# command line by launch_sweep.sh, so they incorporate the deck name.

DECK=$1
if [ -z "$DECK" ]; then
    echo "ERROR: deck name required as first arg, e.g. submit_one.sh c0_0p5"
    exit 1
fi

# --- Software stack -----------------------------------------------------------
module purge
module load apps/binapps/abaqus/2024
module load compilers/intel/17.0.7

# --- Licence guard (auto-requeue if Abaqus tokens are exhausted) -------------
. $ABAQUS_HOME/liccheck.sh

# --- The .inp files live in jobs/; usdfld.f is one level up ------------------
cd "$SLURM_SUBMIT_DIR/jobs"

# --- Memory: track the Slurm allocation so Abaqus does not over- or under-bid -
#   On `multicore`, SLURM_MEM_PER_CPU defaults to 8192 MB.
#   4 cores * 8192 MB = 32768 MB (= 32 GB) available to this job.
MEM=$((SLURM_MEM_PER_CPU*SLURM_NTASKS))

echo "=== $DECK  start $(date) ==="
echo "    cores=$SLURM_NTASKS  mem_per_core=${SLURM_MEM_PER_CPU}MB  mem_total=${MEM}MB"
LC_ALL=C abaqus \
    job=$DECK \
    input=${DECK}.inp \
    user=../usdfld.f \
    cpus=$SLURM_NTASKS \
    scratch=$HOME/scratch \
    memory="$MEM mb" \
    interactive
echo "=== $DECK  end   $(date)  exit=$? ==="
