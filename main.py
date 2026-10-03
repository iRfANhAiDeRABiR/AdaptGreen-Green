"""
Root Launcher for AdaptGreen Simulation Prototype.
Allows executing `python main.py` directly from workspace root.
"""

import sys
from pathlib import Path

# Add adaptgreen to path
BASE_DIR = Path(__file__).resolve().parent
ADAPTGREEN_DIR = BASE_DIR / "adaptgreen"
if str(ADAPTGREEN_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTGREEN_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from adaptgreen.main import run_experiment

if __name__ == "__main__":
    run_experiment()
