"""pytest bootstrap for the lingbot-vla-v2 repository.

The package is normally used from an installed egg / editable checkout; when
pytest runs from a plain clone, make the repo root importable so
``import lingbotvla`` resolves to the working tree.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
