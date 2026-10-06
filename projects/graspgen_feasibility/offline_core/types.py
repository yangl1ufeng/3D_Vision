from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class Scene:
    scene_id: str
    target_id: str
    rgb: np.ndarray
    xyz_world_m: np.ndarray
    valid: np.ndarray
    target: dict
    metadata: dict
    manifest_path: Path
    depth_view: Any = None


@dataclass
class Segmentation:
    mask: np.ndarray
    backend: str
    predicted_quality: float | None
    metadata: dict


@dataclass
class Clouds:
    target_raw_m: np.ndarray
    target_model_m: np.ndarray
    background_full_m: np.ndarray
    background_collision_m: np.ndarray
    outliers_m: np.ndarray
    stats: dict


@dataclass
class Grasps:
    poses_world_m: np.ndarray
    scores: np.ndarray
    metadata: dict

    def __post_init__(self):
        if self.poses_world_m.ndim != 3 or self.poses_world_m.shape[1:] != (4, 4):
            raise ValueError('poses must be Nx4x4')
        if self.scores.shape != (len(self.poses_world_m),):
            raise ValueError('score count must match pose count')
