"""Training Recorder Alias for FlameBox."""
from rescue_env.recorder import (
    ACTION_NAMES,
    FEATURE_NAMES,
    MEMORY_CHANNELS,
    EpisodeRecorder,
    compute_model_hash,
    record_episode,
)

__all__ = [
    "ACTION_NAMES",
    "FEATURE_NAMES",
    "MEMORY_CHANNELS",
    "EpisodeRecorder",
    "compute_model_hash",
    "record_episode",
]
