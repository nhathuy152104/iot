"""Rebuild/replay a completed raw expansion without fitting or reading test."""
import argparse
from pathlib import Path

from raw_expansion.validate import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    validate(parser.parse_args().run_directory)
