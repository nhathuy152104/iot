"""Screen the seven candidates on train, then fit a separate expanded raw arm."""
import argparse
from pathlib import Path

from raw_expansion.experiment import fit, prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume:
        fit(args.resume.resolve())
    else:
        output = prepare()
        if not args.prepare_only:
            fit(output)
