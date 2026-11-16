"""Validate an existing raw-input experiment without fitting or evaluating test."""
import argparse
from pathlib import Path

from raw_branch.validate import validate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    validate(parser.parse_args().run_directory)
