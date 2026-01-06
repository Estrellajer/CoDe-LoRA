import json
import os
from pathlib import Path
from typing import Dict, List, Any

class ManifestManager:
    def __init__(self, manifest_path: str = "artifacts/manifest.json"):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
    def get_base_model_path(self) -> str:
        return self.data.get("base_model_path")
    
    def get_embedding_model(self) -> str:
        return self.data.get("embedding_model")
    
    def get_tasks(self) -> List[str]:
        return self.data.get("tasks", [])
    
    def get_task_config(self, task_name: str) -> str:
        """
        Intelligently get task configuration directory.
        Search priority:
        1. artifacts/runtime_configs/XX_{task_name} (orchestrator-generated runtime config)
        2. {base_configs_dir}/{task_name} (base atomic config)
        3. task_configs mapping in manifest.json (manual override)
        """
        # 1. Try to find from runtime_configs (auto-match prefix)
        runtime_root = Path("artifacts/runtime_configs")
        if runtime_root.exists():
            # Search for directories ending with _task_name
            for item in runtime_root.iterdir():
                if item.is_dir() and item.name.endswith(f"_{task_name}"):
                    return str(item)

        # 2. Try to find from base_configs_dir
        base_dir = self.data.get("base_configs_dir")
        if base_dir:
            task_base_path = Path(base_dir) / task_name
            if task_base_path.exists():
                return str(task_base_path)

        # 3. Fall back to manual definition in manifest
        return self.data.get("task_configs", {}).get(task_name)
    
    def get_prototype_count(self, task_name: str) -> int:
        return self.data.get("prototype_plan", {}).get(task_name, 100)
    
    def get_path(self, key: str) -> str:
        return self.data.get("paths", {}).get(key)
    
    def get_split(self, key: str) -> str:
        return self.data.get("splits", {}).get(key)

    def get_iso_task_dir(self, task_name: str) -> Path:
        """Get task directory under ISO based on task name, supports directories with numeric prefixes."""
        iso_root = Path(self.get_path("iso_root"))
        
        # 1. Try default path with task_ prefix
        default_path = iso_root / f"task_{task_name}"
        if default_path.exists():
            return default_path
            
        # 2. Scan directories with numeric prefixes (e.g., 01_dbpedia)
        if iso_root.exists():
            for item in iso_root.iterdir():
                if item.is_dir() and item.name.endswith(f"_{task_name}"):
                    return item
                    
        # 3. If not found, return predicted path based on order
        tasks = self.get_tasks()
        if task_name in tasks:
            idx = tasks.index(task_name) + 1
            return iso_root / f"{idx:02d}_{task_name}"
            
        return default_path

    def get_iso_adapter_path(self, task_name: str) -> Path:
        return self.get_iso_task_dir(task_name) / "adapter"

    def get_iso_prototypes_dir(self, task_name: str) -> Path:
        return self.get_iso_task_dir(task_name) / "prototypes"

    def get_iso_predictions_dir(self, task_name: str) -> Path:
        return self.get_iso_task_dir(task_name) / "predictions"
    
    def get_shared_step_dir(self, step_k: int) -> Path:
        shared_root = Path(self.get_path("shared_root"))
        return shared_root / f"step_{step_k}"

    def get_shared_final_dir(self) -> Path:
        shared_root = Path(self.get_path("shared_root"))
        return shared_root / "final"

