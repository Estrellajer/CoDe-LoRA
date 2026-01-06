"""Base abstractions for continual LoRA experiments.

This module defines lightweight dataclasses and an abstract base class to
standardize continual-learning implementations across different methods.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskSpec:
    """Metadata for a single task in the sequence."""

    name: str
    dataset_path: str
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    metrics: List[str] = field(default_factory=list)
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalPlan:
    """Evaluation scope after finishing a task."""

    strategy: str = "learned"  # learned | all | custom
    custom_tasks: Optional[List[str]] = None
    metrics_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class RunPaths:
    """Filesystem layout helper.

    Output directory convention:
    runs/<method>/<task_seq>/t{idx}-{task_name}/<timestamp>/
    """

    run_root: str
    method: str
    task_sequence: str
    task_idx: int
    task_name: str
    timestamp: str

    @property
    def task_dir(self) -> str:
        task_folder = f"t{self.task_idx:02d}-{self.task_name}"
        return os.path.join(self.run_root, self.method, self.task_sequence, task_folder, self.timestamp)


@dataclass
class TrainArtifacts:
    """Artifacts and metrics produced by one task's training."""

    checkpoint_path: Optional[str] = None
    predictions_path: Optional[str] = None
    train_metrics: Dict[str, float] = field(default_factory=dict)
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    resource_stats: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class ContinualModel(ABC):
    """Interface every continual-learning variant should implement."""

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any], train_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

    @abstractmethod
    def initialize_or_load(self, checkpoint_path: Optional[str] = None) -> None:
        """Load base model and adapters; optionally warm-start from checkpoint."""

    @abstractmethod
    def get_trainer(self, task: TaskSpec, dataloaders: Dict[str, Any], run_paths: RunPaths) -> Any:
        """Return a trainer instance (e.g. UIETrainer) configured for this task."""

    @abstractmethod
    def evaluate(self, task: TaskSpec, run_paths: RunPaths, split: str = "test") -> Dict[str, float]:
        """Evaluate the model on a specific task and split."""

    def incremental_train(self, task: TaskSpec, dataloaders: Dict[str, Any], run_paths: RunPaths) -> TrainArtifacts:
        """Standard training flow: get trainer, train, and return artifacts."""
        trainer = self.get_trainer(task, dataloaders, run_paths)
        
        train_result = trainer.train()
        
        # Standard save logic
        adapter_path = os.path.join(run_paths.task_dir, "adapter")
        trainer.model.save_pretrained(adapter_path)
        trainer.tokenizer.save_pretrained(adapter_path)
        
        return TrainArtifacts(
            checkpoint_path=adapter_path,
            train_metrics=train_result.metrics,
        )

    @abstractmethod
    def infer(self, batch: Any, task: TaskSpec) -> Any:
        """Specific inference logic (e.g. which adapters to activate)."""

    def save_artifacts(self, artifacts: TrainArtifacts, run_paths: RunPaths) -> None:
        """Persist checkpoints, predictions, and metadata into the task directory."""

        os.makedirs(run_paths.task_dir, exist_ok=True)
        if artifacts.checkpoint_path and os.path.abspath(artifacts.checkpoint_path) != os.path.abspath(run_paths.task_dir):
            # Implement copy/move in concrete subclass if needed.
            pass
        if artifacts.predictions_path and os.path.abspath(artifacts.predictions_path) != os.path.abspath(run_paths.task_dir):
            # Implement copy/move in concrete subclass if needed.
            pass

    def metrics_for_task(self, task: TaskSpec, eval_plan: EvalPlan) -> List[str]:
        """Resolve which metrics to use for this task under the given plan."""

        task_metrics = task.metrics or []
        if eval_plan.metrics_map:
            task_metrics = eval_plan.metrics_map.get(task.name, eval_plan.metrics_map.get("default", task_metrics))
        return task_metrics
