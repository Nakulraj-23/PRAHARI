"""``python -m prahari.models.validator.train``."""

from __future__ import annotations

import json

from prahari.config import get_settings
from prahari.models.validator.model import train

if __name__ == "__main__":
    print(json.dumps(train(get_settings()), indent=1, default=float))
