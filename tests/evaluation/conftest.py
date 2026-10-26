import sys
from pathlib import Path

# Make `src` importable when pytest is launched from any directory.
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
