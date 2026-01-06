"""Factory registry for continual-learning model implementations."""

from __future__ import annotations

from typing import Dict, Type

from .base import ContinualModel


class ModelFactory:
    _registry: Dict[str, Type[ContinualModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class under a string key."""

        def decorator(model_cls: Type[ContinualModel]):
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> ContinualModel:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            raise KeyError(f"Model '{name}' is not registered. Available: [{available}]")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls):
        return sorted(cls._registry)
