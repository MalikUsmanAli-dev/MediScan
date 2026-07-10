"""
auto_install.py
------------------
Best-effort, cross-platform helper that tries to install the Tesseract OCR
system binary automatically using whatever native package manager is
already available (winget on Windows, Homebrew on macOS, apt/dnf on Linux).

This is inherently best-effort: package managers may not be present, the
user may lack permissions, or a step may require an interactive password
prompt we can't supply from inside a Streamlit app. Every code path returns
a structured result with a human-readable log and, on failure, the exact
manual command the user can run themselves.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List


@dataclass
class InstallResult:
    success: bool
    log: List[str] = field(default_factory=list)
    manual_command: str = ""


def _run(cmd: List[str], timeout: int = 300) -> tuple[int, str]:
    """Run a command non-interactively, capturing combined output."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
        return proc.returncode, proc.stdout or ""
    except FileNotFoundError:
        return 127, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "Command timed out."
    except Exception as exc:  # noqa: BLE001
        return 1, f"Unexpected error running {cmd[0]}: {exc}"


def attempt_auto_install() -> InstallResult:
    """
    Try to install Tesseract automatically for the current OS.
    Never raises — always returns an InstallResult so the UI can display
    the log and fall back to manual instructions if needed.
    """
    system = platform.system()
    log: List[str] = [f"Detected OS: {system}"]

    if system == "Windows":
        return _install_windows(log)
    if system == "Darwin":
        return _install_macos(log)
    if system == "Linux":
        return _install_linux(log)

    log.append("Unrecognized OS — automatic install isn't supported here.")
    return InstallResult(success=False, log=log, manual_command="See the Settings page for manual install links.")


# --------------------------------------------------------------------------- #
def _install_windows(log: List[str]) -> InstallResult:
    manual = r"winget install --id UB-Mannheim.TesseractOCR -e"
    if shutil.which("winget") is None:
        log.append("winget not found on this system (needs Windows 10 1809+ / Windows 11 with App Installer).")
        return InstallResult(success=False, log=log, manual_command=manual)

    log.append("winget found. Attempting silent install of Tesseract-OCR...")
    code, output = _run(
        [
            "winget", "install", "--id", "UB-Mannheim.TesseractOCR", "-e",
            "--silent", "--accept-package-agreements", "--accept-source-agreements",
        ],
        timeout=600,
    )
    log.append(output.strip() or f"(no output, exit code {code})")
    if code == 0:
        log.append("winget reported success. You may need to restart the app for PATH changes to apply.")
        return InstallResult(success=True, log=log)

    log.append(f"winget exited with code {code}.")
    return InstallResult(success=False, log=log, manual_command=manual)


def _install_macos(log: List[str]) -> InstallResult:
    manual = "brew install tesseract"
    if shutil.which("brew") is None:
        log.append("Homebrew not found. Install it from https://brew.sh first, then re-run.")
        return InstallResult(success=False, log=log, manual_command=manual)

    log.append("Homebrew found. Running: brew install tesseract")
    code, output = _run(["brew", "install", "tesseract"], timeout=600)
    log.append(output.strip() or f"(no output, exit code {code})")
    if code == 0:
        return InstallResult(success=True, log=log)

    log.append(f"brew exited with code {code}.")
    return InstallResult(success=False, log=log, manual_command=manual)


def _install_linux(log: List[str]) -> InstallResult:
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0

    if shutil.which("apt-get") is not None:
        manual = "sudo apt-get update && sudo apt-get install -y tesseract-ocr"
        if not is_root:
            log.append(
                "apt-get needs root privileges and Streamlit can't prompt for your sudo "
                "password interactively — please run the command below in a terminal instead."
            )
            return InstallResult(success=False, log=log, manual_command=manual)
        log.append("Running: apt-get update && apt-get install -y tesseract-ocr")
        code1, out1 = _run(["apt-get", "update"], timeout=300)
        log.append(out1.strip())
        code2, out2 = _run(["apt-get", "install", "-y", "tesseract-ocr"], timeout=600)
        log.append(out2.strip())
        if code2 == 0:
            return InstallResult(success=True, log=log)
        return InstallResult(success=False, log=log, manual_command=manual)

    if shutil.which("dnf") is not None:
        manual = "sudo dnf install -y tesseract"
        if not is_root:
            log.append("dnf needs root privileges — please run the command below in a terminal.")
            return InstallResult(success=False, log=log, manual_command=manual)
        code, output = _run(["dnf", "install", "-y", "tesseract"], timeout=600)
        log.append(output.strip())
        return InstallResult(success=(code == 0), log=log, manual_command=manual)

    log.append("No supported package manager (apt-get/dnf) detected.")
    return InstallResult(success=False, log=log, manual_command="Install 'tesseract-ocr' using your distro's package manager.")
