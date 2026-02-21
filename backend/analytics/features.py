"""Feature engineering helpers.

We keep these functions separate so you can test / tweak thresholds fast.
"""

import pandas as pd
from .bias_rules import add_features  # re-export for convenience

__all__ = ["add_features", "pd"]
