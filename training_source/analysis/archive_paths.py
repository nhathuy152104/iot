"""Resolve relocated archive paths without rewriting frozen provenance files."""
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = PureWindowsPath(r'D:\CODE\IOT training')


def resolve_archive_path(value, root=ROOT):
    path = PureWindowsPath(value)
    try:
        relative = path.relative_to(LEGACY_ROOT)
    except ValueError:
        result = Path(value)
    else:
        if '..' in relative.parts:
            raise ValueError('Parent traversal in legacy archive path')
        result = Path(root).joinpath(*relative.parts)
    if not result.exists():
        raise FileNotFoundError(f'Archive path unavailable: {value} -> {result}')
    return result
