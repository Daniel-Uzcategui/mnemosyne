#!/usr/bin/env bash

set -euo pipefail

# Example backend startup script for the router manager.
# It mirrors the real production flow more closely than the old minimal example:
# - binds the expected HTTP port
# - enables metrics
# - keeps KV snapshots in the configured slot-save path
# - preloads the target model through /models/load

HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-11434}"
GPU_GROUP="${CUDA_VISIBLE_DEVICES:-2,3}"
MODEL_NAME="${LLAMA_MODEL_NAME:-qwen}"
MODELS_PRESET="${LLAMA_MODELS_PRESET:-./models.ini}"
SLOT_SAVE_PATH="${LLAMA_SLOT_SAVE_PATH:-$HOME/Models/ramdisk_cache}"
STARTUP_WAIT_SECONDS="${LLAMA_STARTUP_WAIT_SECONDS:-3}"
REASONING_BUDGET="${LLAMA_REASONING_BUDGET:-4096}"
REASONING_BUDGET_MESSAGE="${LLAMA_REASONING_BUDGET_MESSAGE:-[SYSTEM ALERT: Reasoning budget exceeded. Stop immediately and ask the user for guidance.]}"

export CUDA_VISIBLE_DEVICES="$GPU_GROUP"

llama-server --models-preset "$MODELS_PRESET" \
  --port "$PORT" \
  --host "$HOST" \
  -b 2048 \
  -ub 2048 \
  -sm layer \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -fa 1 \
  -np 1 \
  --metrics \
  --no-context-shift \
  --no-mmap \
  --slot-save-path "$SLOT_SAVE_PATH" \
  --reasoning on \
  --reasoning-budget "$REASONING_BUDGET" \
  --reasoning-budget-message "$REASONING_BUDGET_MESSAGE" \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  "$@" &

sleep "$STARTUP_WAIT_SECONDS"

curl -fsS -X POST "http://127.0.0.1:${PORT}/models/load" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"${MODEL_NAME}\"}" > /dev/null

wait
