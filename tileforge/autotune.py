"""Autotuning engine for TileForge kernels."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple


class Config:
    """Represents a kernel execution configuration (e.g. tile sizes)."""
    def __init__(self, kwargs: Dict[str, Any], num_warps: int = 4, num_stages: int = 2):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages

    def __repr__(self) -> str:
        return f"Config({self.kwargs}, num_warps={self.num_warps}, num_stages={self.num_stages})"


class AutotunedKernel:
    """Wrapper around kernel function that autotunes execution configs."""
    def __init__(self, fn: Callable, configs: List[Config], key: Optional[List[str]] = None):
        self.fn = fn
        self.configs = configs
        self.key = key or []
        self.best_config_cache: Dict[Tuple, Config] = {}

    def __call__(self, *args, **kwargs):
        cache_key = tuple(kwargs.get(k) for k in self.key if k in kwargs)
        if cache_key in self.best_config_cache:
            best_cfg = self.best_config_cache[cache_key]
            return self.fn(*args, **{**kwargs, **best_cfg.kwargs})

        if not self.configs:
            return self.fn(*args, **kwargs)

        best_time = float("inf")
        best_cfg = self.configs[0]

        for cfg in self.configs:
            try:
                self.fn(*args, **{**kwargs, **cfg.kwargs})
                t0 = time.perf_counter()
                for _ in range(5):
                    self.fn(*args, **{**kwargs, **cfg.kwargs})
                t1 = time.perf_counter()
                avg_time = (t1 - t0) / 5.0

                if avg_time < best_time:
                    best_time = avg_time
                    best_cfg = cfg
            except Exception:
                continue

        if cache_key:
            self.best_config_cache[cache_key] = best_cfg

        return self.fn(*args, **{**kwargs, **best_cfg.kwargs})


def autotune(configs: List[Config], key: Optional[List[str]] = None) -> Callable:
    """Decorator to automatically autotune kernel execution over candidate configs."""
    def decorator(fn: Callable) -> AutotunedKernel:
        return AutotunedKernel(fn, configs=configs, key=key)
    return decorator
