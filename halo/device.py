"""Compute-device selection.

Local development runs on the Mac (MPS). The quantized generation in
Phase 2 needs CUDA, because bitsandbytes has no MPS or CPU backend, so
that is guarded explicitly rather than failing deep inside a call.
"""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    """Return the fastest torch device available on this machine.

    Returns
    -------
    torch.device
        ``cuda`` if present, else ``mps``, else ``cpu``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cuda_required(feature: str) -> torch.device:
    """Return the CUDA device, or explain why the step cannot run.

    Parameters
    ----------
    feature : str
        Name of the feature that needs CUDA, used in the error message.

    Returns
    -------
    torch.device
        The CUDA device.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{feature} needs a CUDA GPU: bitsandbytes has no MPS or CPU "
            f"backend. Run this step on Colab or the lab GPU."
        )
    return torch.device("cuda")
