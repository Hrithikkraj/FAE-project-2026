"""Start the live app without setting PYTHONPATH: python scripts/run_app.py [--wav FILE]."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swara_aware.app.main import main  # noqa: E402

if __name__ == "__main__":
    main()
