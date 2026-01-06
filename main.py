"""Entry point to run the continual orchestrator."""

import argparse

from src.continual.orchestrator import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Continual LoRA runner")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    args = parser.parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
