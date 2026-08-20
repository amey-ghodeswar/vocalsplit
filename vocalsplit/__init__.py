"""vocalsplit -- vocal/instrument separation that knows when not to separate."""
from .decide import Thresholds, classify
from .features import stem_metrics
from .pipeline import Config, analyze, write_outputs
from .separator import DemucsSeparator

__version__ = "0.1.0"
__all__ = [
    "Config", "DemucsSeparator", "Thresholds",
    "analyze", "classify", "stem_metrics", "write_outputs",
]
