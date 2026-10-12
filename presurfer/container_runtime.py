"""SPM container runtime commands and execution."""

import os
import shutil
import subprocess
from datetime import datetime, UTC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from presurfer.output_paths import container_path

IMAGE = "ghcr.io/spm/spm-docker:docker-matlab-25.01.02"
SIF_IMAGE = "spm.sif"
Runtime = Literal["docker", "apptainer"]

__all__ = [
    "IMAGE",
    "SIF_IMAGE",
    "Container",
    "Runtime",
    "check_container",
    "container_command",
    "run_spm_batch",
]


@dataclass(frozen=True)
class Container:
    """A container runtime together with the image it runs.

    Attributes:
        runtime: ``"docker"`` or ``"apptainer"``.
        image: Docker image tag, or the path to a local Apptainer image.

    Raises:
        RuntimeError: If ``runtime`` is not supported.
    """

    runtime: Runtime
    image: str

    def __post_init__(self) -> None:
        """Reject unsupported runtimes as soon as the container is created."""
        if self.runtime not in ("docker", "apptainer"):
            raise RuntimeError(
                f"Unsupported runtime: {self.runtime}. Use 'docker' or 'apptainer'."
            )


def _require_docker_image(image: str) -> None:
    """Check that Docker is installed and the image has been pulled.

    Args:
        image: Docker image tag.

    Raises:
        RuntimeError: If Docker is not on PATH or the image is not available
            locally.
    """
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker was not found on PATH. Install Docker, then pull the SPM image."
        )
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if inspect.returncode:
        raise RuntimeError(
            f"SPM image is not available locally: {image}\nPull it first: docker pull {image}"
        )


def _apptainer_executable(image: str) -> str:
    """Find Apptainer and check that the image file exists.

    Args:
        image: Path to a local Apptainer image.

    Returns:
        Path to the ``apptainer`` executable.

    Raises:
        RuntimeError: If Apptainer is not on PATH or the image file is missing.
    """
    executable = shutil.which("apptainer")
    if executable is None:
        raise RuntimeError("Apptainer was not found on PATH.")
    if not Path(image).is_file():
        raise RuntimeError(
            f"Apptainer image does not exist: {image}\n"
            f"Pull it first: apptainer pull --name {SIF_IMAGE} "
            "oras://ghcr.io/spm/spm-docker:singularity-matlab-25.01.02"
        )
    return executable


def container_command(
    mount_root: Path, job_file: Path, container: Container
) -> list[str]:
    """Build a Docker or Apptainer SPM batch command.

    Args:
        mount_root: Host directory mounted at ``/data``.
        job_file: Generated SPM batch script below ``mount_root``.
        container: Runtime and image to run.

    Returns:
        Command arguments suitable for ``subprocess.run``.

    Raises:
        RuntimeError: If Apptainer is unavailable or the image file does not
            exist.
    """
    job = container_path(job_file, mount_root)
    if container.runtime == "docker":
        return [
            "docker",
            "run",
            "--rm",
            #"--user",
            #f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{mount_root.resolve()}:/data",
            container.image,
            "batch",
            job,
        ]
    return [
        _apptainer_executable(container.image),
        "exec",
        "--bind",
        f"{mount_root.resolve()}:/data",
        container.image,
        "spm",
        "batch",
        job,
    ]


def check_container(container: Container) -> str:
    """Check that a selected SPM container can start and report its version.

    Args:
        container: Runtime and image to check.

    Returns:
        Human-readable runtime, image, and SPM version information.

    Raises:
        RuntimeError: If the runtime or image is unavailable, or SPM cannot
            start inside the selected container.
    """
    if container.runtime == "docker":
        _require_docker_image(container.image)
        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            container.image,
            "--version",
        ]
        label = "Docker image"
    else:
        executable = _apptainer_executable(container.image)
        command = [executable, "exec", container.image, "--version"]
        label = f"{Path(executable).name.capitalize()} image"

    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        message = f"SPM container check failed (exit status {result.returncode})."
        raise RuntimeError(f"{message}\n{detail}" if detail else message)
    version = result.stdout.strip()
    return f"Runtime: {container.runtime}\n{label}: {container.image}\nSPM version:\n{version}"


def run_spm_batch(
    mount_root: Path,
    job_file: Path,
    container: Container,
    log_path: Path,
    threads: int | None = None,
) -> None:
    """Run a generated SPM batch in the selected container runtime.

    Each call appends a section to the log that records the container, image,
    and MATLAB thread count, followed by the captured output. Thread count can
    change results in the last digits, so the log lets a run be reproduced.

    Args:
        mount_root: Host directory mounted at ``/data``.
        job_file: Generated SPM batch script below ``mount_root``.
        container: Runtime and image to run.
        log_path: File that receives the captured standard output and error.
        threads: MATLAB thread count written into the batch script, recorded
            in the log. ``None`` means the MATLAB default.

    Raises:
        RuntimeError: If the runtime or image is unavailable, or the SPM batch
            exits unsuccessfully.
    """
    if container.runtime == "docker":
        _require_docker_image(container.image)
    command = container_command(mount_root, job_file, container)
    header = (
        f"SPM batch: {job_file.name}\n"
        f"Container: {container.runtime} {container.image}\n"
        f"Command: {' '.join(command)}\n"
        f"Root: {mount_root}\n"
        f"MATLAB threads: {threads if threads is not None else 'MATLAB default'}\n\n"
        f"Start(UTC): {datetime.now(UTC).strftime('%y%m%dT%H:%M:%S')}\n"
    )
    if os.environ.get("VERBOSE"):
        print("Saving log to '{log_path}'")
        print(header)

    # open log file early for early feedback. Command may take a while to run.
    with log_path.open("a") as log:
        log.write(header)

    # TODO: stream command output to log file
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    with log_path.open("a") as log:
        log.write("STDOUT\n------\n"
            + result.stdout
            + "\nSTDERR\n------\n"
            + result.stderr
            + "\nExit Code: {result.returncode}"
            + f"\nFinish(UTC): {datetime.now(UTC).strftime('%y%m%dT%H:%M:%S')}"
            + "\n"
        )
    if result.returncode:
        raise RuntimeError(
            f"SPM batch failed ({container.runtime} exit status {result.returncode}). "
            f"See {log_path}."
        )
