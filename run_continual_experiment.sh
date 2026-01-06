#!/bin/bash
# Generic script to run continual learning experiments
# 
# Usage:
#   bash run_continual_experiment.sh <config_file_path>
#
# Examples:
#   bash run_continual_experiment.sh configs/experiments/llama/order1_codelora_llama_continual.yaml
#   bash run_continual_experiment.sh configs/experiments/llama/order2_nlora_llama_continual.yaml

set -e  # Exit on error

# Check if config file argument is provided
if [ -z "$1" ]; then
    echo "Error: Please provide a config file path"
    echo "Usage: bash run_continual_experiment.sh <config_file_path>"
    echo ""
    echo "Available config examples:"
    echo "  - configs/experiments/llama/order1_codelora_llama_continual.yaml"
    echo "  - configs/experiments/llama/order1_colora_llama_continual.yaml"
    echo "  - configs/experiments/llama/order1_nlora_llama_continual.yaml"
    echo "  - configs/experiments/llama/order1_olora_llama_continual.yaml"
    echo "  - configs/experiments/llama/order1_delora_llama_continual.yaml"
    exit 1
fi

CONFIG_FILE="$1"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Get script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH=$PYTHONPATH:$PWD/src

# Activate virtual environment managed by uv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif command -v uv &> /dev/null; then
    echo "Creating virtual environment with uv..."
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
else
    echo "Warning: Virtual environment not found and uv is not available"
    echo "Please install uv or create a virtual environment manually"
fi

# Extract experiment info from config filename (for log path)
# Example: configs/experiments/llama/order1_codelora_llama_continual.yaml
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .yaml)
# Extract order and method
if [[ $CONFIG_BASENAME =~ order([0-9]+)_([a-z]+)_llama_continual ]]; then
    ORDER="${BASH_REMATCH[1]}"
    METHOD="${BASH_REMATCH[2]}"
    LOG_DIR="logs_and_outputs_llama/order_${ORDER}/logs"
    LOG_FILE="${LOG_DIR}/${METHOD}_train.log"
else
    # Fallback to default path if parsing fails
    LOG_DIR="logs_and_outputs/logs"
    LOG_FILE="${LOG_DIR}/experiment_$(date +%Y%m%d_%H%M%S).log"
    echo "Warning: Could not parse experiment info from config filename, using default log path: $LOG_FILE"
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Generate random port to avoid conflicts (for multi-GPU training)
if command -v shuf &> /dev/null; then
    port=$(shuf -i25000-30000 -n1)
elif command -v jot &> /dev/null; then
    port=$(jot -r 1 25000 30000)
else
    port=$((25000 + RANDOM % 5000))
fi

echo "=========================================="
echo "Starting Continual Learning Experiment"
echo "Config file: $CONFIG_FILE"
echo "Log directory: $LOG_DIR"
echo "Master port: $port"
echo "=========================================="

# Run experiment
deepspeed --master_port $port main.py --config "$CONFIG_FILE" \
    > "$LOG_FILE" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Experiment completed! Log file: $LOG_FILE"
    echo "=========================================="
else
    echo "=========================================="
    echo "Experiment failed with exit code: $EXIT_CODE"
    echo "Please check the log file: $LOG_FILE"
    echo "=========================================="
    exit $EXIT_CODE
fi
