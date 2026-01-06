# CoDe-LoRA: Resolving the Orthogonality Paradox in Continual Learning of LLMs via Knowledge Consolidation and Decoupling

This repository contains the implementation of CoDe-LoRA, a method for continual learning with language models. The codebase supports multiple LoRA variants including CoDe-LoRA, Co-LoRA, N-LoRA, O-LoRA, and De-LoRA.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and virtual environment creation.

### Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Create Environment and Install Dependencies

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Running Experiments

### Basic Usage

```bash
bash run_continual_experiment.sh <config_file_path>
```

### Examples

Run CoDe-LoRA with Order 1 task sequence:

```bash
bash run_continual_experiment.sh configs/experiments/llama/order1_codelora_llama_continual.yaml
```

Run Co-LoRA with Order 2 task sequence:

```bash
bash run_continual_experiment.sh configs/experiments/llama/order2_colora_llama_continual.yaml
```

## Available Configurations

All configuration files are located in `configs/experiments/llama/` with the naming pattern:

```
order<number>_<method>_llama_continual.yaml
```

### Supported Methods

- `codelora` - CoDe-LoRA
- `colora` - Co-LoRA
- `nlora` - N-LoRA
- `olora` - O-LoRA
- `delora` - De-LoRA

### Task Sequences

- `order1`, `order2`, `order3` - Different task ordering configurations

### Switching Configurations

Simply specify a different config file:

```bash
# Change method
bash run_continual_experiment.sh configs/experiments/llama/order1_colora_llama_continual.yaml

# Change task sequence
bash run_continual_experiment.sh configs/experiments/llama/order2_codelora_llama_continual.yaml
```

## Output

Logs are automatically saved to:
- `logs_and_outputs_llama/order_<number>/logs/<method>_train.log`

## Requirements

- Python 3.8+
- CUDA-capable GPU
- DeepSpeed
- See `requirements.txt` for full dependencies
