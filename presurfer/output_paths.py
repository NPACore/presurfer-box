"""NIfTI input and run-directory path helpers."""

import gzip
import shutil
from datetime import UTC, datetime
from pathlib import Path
import os

__all__ = ["container_path", "materialise_input", "nii_stem", "utc_run_directory"]


def utc_run_directory(parent: Path, temporary: bool = False) -> Path:
    """Create a collision-safe UTC timestamped workflow directory.

    Args:
        parent: Directory that receives the new run directory.
        temporary: Whether to prefix the name with ``tmp_`` for a stage that
            the workflow removes afterwards.

    Returns:
        Path to the new directory. A numeric suffix is added if a directory
        with the same timestamp already exists.
    """
    prefix = "tmp_" if temporary else ""
    candidate = parent / f"{prefix}{datetime.now(UTC).strftime('%y%m%dT%H:%M:%S')}_presurfer"
    #: PRESURFER_FOLDER overwrites dynamic output generation. Use what user specifies
    candidate = Path(os.environ.get('PRESURFER_FOLDER',candidate))

    # skip or remove
    if candidate.exists():
        if os.environ.get("REDO"):
            print(f"WARNING: removing {candidate} b/c REDO in environment")
            shutil.rmtree(candidate)
        else:
            # TODO: exit without error for existing if final file
            raise Exception(f"# output folder {candidate} already exists. export REDO=1 to remove and redo.")

    candidate.mkdir()
    return candidate


def nii_stem(path: Path) -> str:
    """Return a NIfTI basename without ``.nii`` or ``.nii.gz``.

    Args:
        path: NIfTI file path.

    Returns:
        Filename without its NIfTI extension.

    Raises:
        RuntimeError: If ``path`` does not have a supported NIfTI extension.
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise RuntimeError(f"Expected a .nii or .nii.gz file, got: {path}")


def materialise_input(source: Path, output_dir: Path) -> Path:
    """Copy a NIfTI input into an output directory.

    Compressed inputs are decompressed to ``.nii`` without modifying the
    original source file.

    Args:
        source: Existing ``.nii`` or ``.nii.gz`` input file.
        output_dir: Directory that receives the materialised ``.nii`` file.

    Returns:
        Path to the copied or decompressed ``.nii`` file.

    Raises:
        RuntimeError: If ``source`` does not exist or is not a NIfTI file.
    """
    if not source.is_file():
        raise RuntimeError(f"Input file does not exist: {source}")
    target = output_dir / f"{nii_stem(source)}.nii"
    if source.name.endswith(".nii.gz"):
        with gzip.open(source, "rb") as compressed, target.open("wb") as plain:
            shutil.copyfileobj(compressed, plain)
    else:
        shutil.copy2(source, target)
    return target


def container_path(host_path: Path, mount_root: Path) -> str:
    """Map a path below a host mount directory to its container path.

    Args:
        host_path: Host path to expose inside the container.
        mount_root: Host directory mounted at ``/data``.

    Returns:
        Equivalent absolute path below ``/data`` inside the container.

    Raises:
        RuntimeError: If ``host_path`` is outside ``mount_root``.
    """
    try:
        return "/data/" + str(host_path.resolve().relative_to(mount_root.resolve()))
    except ValueError:
        raise RuntimeError(
            f"Path is outside the mounted directory {mount_root}: {host_path}"
        ) from None
