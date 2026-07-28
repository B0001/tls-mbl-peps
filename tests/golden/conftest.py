"""tests/golden may import `prototypes/` as an independent oracle (ARCHITECTURE.md §4;
never from src/). The prototypes import each other flatly, so their directory goes on
sys.path here, quarantined to this test package."""

import sys
from pathlib import Path

PROTOTYPES = Path(__file__).resolve().parents[2] / "prototypes"
if str(PROTOTYPES) not in sys.path:
    sys.path.insert(0, str(PROTOTYPES))
