#!/bin/bash
#SBATCH --job-name=base_grpo_ext_shaped_t06_ms700_lr1e5_n4_r64_minrem1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=54G
#SBATCH --partition=l40
#SBATCH --output=/path/to/MasterThesis/base_grpo_ext/eval_%x_%j.log
#SBATCH --chdir=/path/to/MasterThesis/base_grpo_ext
#SBATCH --gres=gpu:1

PROJECT=/path/to/MasterThesis/base_grpo_ext
RUN_DIR=$PROJECT/runs/base_grpo_ext
SCRIPT=run_evaluation_Nemo-full_base_grpo_ext.py

# ---- every run works in its own folder -> runs can be queued / run in parallel ----
# The folder gets a SNAPSHOT of the code (editing the .py files later does not affect this run)
# and its own unsloth_compiled_cache (Unsloth writes it in the working directory at every start).
# The model in pretrained_models/ and the data are only READ, so they are shared safely.
cd "$PROJECT" || exit 1
case "$RUN_DIR" in "$PROJECT"/runs/?*) ;; *) echo "bad RUN_DIR"; exit 1;; esac   # safety for rm -rf
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp "$PROJECT"/*.py "$RUN_DIR"/
ln -s "$PROJECT/input" "$RUN_DIR/input"
ln -s "$PROJECT/train_tasks.json" "$RUN_DIR/train_tasks.json"
ln -s "$PROJECT/held_out_tasks.json" "$RUN_DIR/held_out_tasks.json"
cd "$RUN_DIR" || exit 1
echo "=== working in $RUN_DIR ==="

export HF_HOME="/path/to/cache/huggingface"
export PYTHONPATH="$RUN_DIR/"
export TRITON_CACHE_DIR="/path/to/cache/triton"
unset RUN_TAG    # outputs go directly into $RUN_DIR (the current folder)

# Fix PyTorch CUDA VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start background VRAM monitoring (Logs VRAM every 5 seconds)
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > vram_log.txt &
MONITOR_PID=$!

source /path/to/venv/bin/activate
PY=/path/to/venv/bin/python3

# Sequential Shard Loop
N_SHARDS=4
for SHARD_ID in $(seq 0 $((N_SHARDS-1))); do
    MAX_ATTEMPTS=8
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        echo "=== Shard $SHARD_ID/$N_SHARDS, Attempt $attempt/$MAX_ATTEMPTS ==="
        $PY $SCRIPT $SHARD_ID $N_SHARDS
        status=$?
        if [ $status -eq 0 ]; then
            echo "=== Shard $SHARD_ID completed successfully ==="
            break
        fi
        sleep 10
    done
done

# Merge all shards into final output
$PY merge_submissions.py $N_SHARDS .

# Stop the background monitor process
kill $MONITOR_PID
