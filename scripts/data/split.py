#!/usr/bin/env python3
"""Balance, filter and split the dataset; see python -m scripts.data.split --help."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.create_split import main

if __name__ == '__main__':
    main()
