#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@example.com
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=54G
#SBATCH --partition=l40
#SBATCH --output=/path/to/MasterThesis/final_test/eval_%x_%j.log
#SBATCH --chdir=/path/to/MasterThesis/final_test
#SBATCH --gres=gpu:1

# One sbatch for experiments 4, 5, 8. Choose the run on the command line, e.g.:
#   sbatch --job-name=exp4_raw_ext      --export=ALL,GRPO_BASE=raw,GRPO_REWARD=ext      run_grpo_final.sbatch
#   sbatch --job-name=exp5_raw_ext_int  --export=ALL,GRPO_BASE=raw,GRPO_REWARD=ext_int  run_grpo_final.sbatch
#   sbatch --job-name=exp8_sft_int_only --export=ALL,GRPO_BASE=sft,GRPO_REWARD=int_only run_grpo_final.sbatch
# qsigned extrinsic reward (no fixed terms) and intrinsic scales for the sensitivity runs, e.g.:
#   sbatch --job-name=q_sft_ext     --export=ALL,GRPO_BASE=sft,GRPO_REWARD=ext,GRPO_EXT=qsigned     run_grpo_final.sbatch
#   sbatch --job-name=q_raw_ext     --export=ALL,GRPO_BASE=raw,GRPO_REWARD=ext,GRPO_EXT=qsigned     run_grpo_final.sbatch
#   sbatch --job-name=q_sft_ext_int --export=ALL,GRPO_BASE=sft,GRPO_REWARD=ext_int,GRPO_EXT=qsigned run_grpo_final.sbatch
#   sbatch --job-name=v4b_a01       --export=ALL,GRPO_BASE=sft,GRPO_REWARD=ext_int,CONF_ALPHA=0.1   run_grpo_final.sbatch
# Second seed:      add SEED=43
# G=8 memory test:  sbatch --job-name=memtest_g8 --export=ALL,GRPO_BASE=sft,GRPO_REWARD=ext_int,GRPO_EXT=qsigned,GRPO_G=8,GRPO_MEMTEST=1,GRPO_MAX_STEPS=10 run_grpo_final.sbatch
# Quick test (50 steps, shard 0 only):
#   sbatch --job-name=test_raw_ext --export=ALL,GRPO_BASE=raw,GRPO_REWARD=ext,GRPO_MAX_STEPS=50,SHARDS="0" run_grpo_final.sbatch

: "${GRPO_BASE:?set GRPO_BASE}" "${GRPO_REWARD:?set GRPO_REWARD}"
export GRPO_BASE GRPO_REWARD
export GRPO_EXT=${GRPO_EXT:-shaped} CONF_ALPHA=${CONF_ALPHA:-0.3} CUR_BETA=${CUR_BETA:-0.2}
export SEED=${SEED:-42} GRPO_G=${GRPO_G:-4} GRPO_MEMTEST=${GRPO_MEMTEST:-0}
export GRPO_MAX_STEPS=${GRPO_MAX_STEPS:-700}
SHARDS=${SHARDS:-"0 1 2 3"}
N_SHARDS=4

PROJECT=/path/to/MasterThesis/final_test
SUFFIX=""
[ "$SEED" != "42" ] && SUFFIX="${SUFFIX}_seed${SEED}"
[ "$GRPO_G" != "4" ] && SUFFIX="${SUFFIX}_g${GRPO_G}"
[ "$GRPO_MEMTEST" = "1" ] && SUFFIX="${SUFFIX}_memtest"
RUN_DIR=$PROJECT/runs/final_${GRPO_BASE}_${GRPO_REWARD}_${GRPO_EXT}_a${CONF_ALPHA}_b${CUR_BETA}_ms${GRPO_MAX_STEPS}${SUFFIX}
SCRIPT=run_evaluation_Nemo-full_grpo_final.py

# ---- every run works in its own folder (snapshot of the code, own unsloth cache) ----
cd "$PROJECT" || exit 1
case "$RUN_DIR" in "$PROJECT"/runs/?*) ;; *) echo "bad RUN_DIR"; exit 1;; esac   # safety for rm -rf
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp "$PROJECT"/*.py "$RUN_DIR"/
ln -s "$PROJECT/input" "$RUN_DIR/input"
ln -s "$PROJECT/train_tasks.json" "$RUN_DIR/train_tasks.json"
ln -s "$PROJECT/held_out_tasks.json" "$RUN_DIR/held_out_tasks.json"
cd "$RUN_DIR" || exit 1
echo "=== working in $RUN_DIR (GRPO_BASE=$GRPO_BASE GRPO_REWARD=$GRPO_REWARD ext=$GRPO_EXT alpha=$CONF_ALPHA beta=$CUR_BETA steps=$GRPO_MAX_STEPS shards=$SHARDS) ==="

export HF_HOME="/path/to/cache/huggingface"
export PYTHONPATH="$RUN_DIR/"
export TRITON_CACHE_DIR="/path/to/cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
unset RUN_TAG    # outputs go directly into $RUN_DIR

nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu --format=csv -l 5 > vram_log.txt &
MONITOR_PID=$!

PY=/path/to/venv/bin/python3    # new venv (works after the Debian 13 upgrade)

for SHARD_ID in $SHARDS; do
    MAX_ATTEMPTS=8; [ "$GRPO_MEMTEST" = "1" ] && MAX_ATTEMPTS=1     # memory test: an OOM must not be retried
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
        echo "=== Shard $SHARD_ID/$N_SHARDS, Attempt $attempt/$MAX_ATTEMPTS ==="
        $PY $SCRIPT $SHARD_ID $N_SHARDS && { echo "=== Shard $SHARD_ID completed successfully ==="; break; }
        sleep 10
    done
done

[ "$SHARDS" = "0 1 2 3" ] && [ "$GRPO_MEMTEST" != "1" ] && $PY merge_submissions.py $N_SHARDS .

kill $MONITOR_PID
