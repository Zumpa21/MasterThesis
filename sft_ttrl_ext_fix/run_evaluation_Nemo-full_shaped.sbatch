#!/bin/bash
#SBATCH --job-name=shaped_ext_fix_sft_ttrl_t06_ms700_lr2e5_n4_r64_minrem1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=54G
#SBATCH --partition=l40
#SBATCH --output=/path/to/MasterThesis/sft_ttrl_ext_fix/eval_%x_%j.log
#SBATCH --chdir=/path/to/MasterThesis/sft_ttrl_ext_fix
#SBATCH --gres=gpu:1

export HF_HOME="/path/to/cache/huggingface"
export PYTHONPATH="/path/to/MasterThesis/sft_ttrl_ext_fix/"
export TRITON_CACHE_DIR="/path/to/cache/triton"

# Fix PyTorch CUDA VRAM fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Start background VRAM monitoring (Logs VRAM every 5 seconds to vram_log.txt)
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
        $PY run_evaluation_Nemo-full_shaped.py $SHARD_ID $N_SHARDS
        status=$?
        if [ $status -eq 0 ]; then
            echo "=== Shard $SHARD_ID completed successfully ==="
            break
        fi
        sleep 10
    done
done

# Merge all shards into final output
$PY merge_submissions.py $N_SHARDS

# Stop the background monitor process
kill $MONITOR_PID
