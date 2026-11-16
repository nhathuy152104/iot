"""Run a legacy analysis entry point with an in-memory relocated JSON path view.

Frozen scripts and provenance bytes remain unchanged, so checksum checks still
apply. The entry point retains its normal behavior (including writes/fitting);
this is NOT a sandbox or a read-only validator.
"""
import json
from pathlib import Path, PureWindowsPath
import runpy
import sys
from unittest.mock import patch

from archive_paths import LEGACY_ROOT, ROOT


def relocate_values(value):
    if isinstance(value, dict):
        return {relocate_values(k): relocate_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate_values(v) for v in value]
    if isinstance(value, str):
        try:
            relative = PureWindowsPath(value).relative_to(LEGACY_ROOT)
        except ValueError:
            return value
        if '..' in relative.parts:
            raise ValueError('Parent traversal in legacy archive path')
        return str(ROOT.joinpath(*relative.parts))
    return value


def main():
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python run_relocated.py <analysis-script.py> [arguments]')
    script = Path(sys.argv[1]).resolve()
    if script.parent != Path(__file__).resolve().parent or script == Path(__file__).resolve():
        raise SystemExit('Only analysis entry points in this directory are supported')
    original = json.loads

    def relocated_loads(*args, **kwargs):
        return relocate_values(original(*args, **kwargs))

    args = sys.argv
    try:
        sys.argv = [str(script), *args[2:]]
        with patch.object(json, 'loads', relocated_loads):
            runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv = args


if __name__ == '__main__':
    main()
