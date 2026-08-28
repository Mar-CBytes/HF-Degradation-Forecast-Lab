"""
Pipeline __init__ – expose convenient run_pipeline() entry point.
"""
from .ingest    import load_and_unify, save_unified
from .label     import label_dataset, run as run_labeling
from .features  import engineer_features, run as run_features
from .split     import split_and_save, get_split_boundaries
from .train     import train
from .preprocess import build_feature_vector, load_feature_meta

__all__ = [
    "load_and_unify", "save_unified",
    "label_dataset",  "run_labeling",
    "engineer_features", "run_features",
    "split_and_save", "get_split_boundaries",
    "train",
    "build_feature_vector", "load_feature_meta",
]
