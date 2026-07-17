"""Opt-in equivalence test against the upstream Pyro GPLFR.

Skipped unless ``GPLFR_UPSTREAM_DIR`` points at a clone of
https://github.com/edstevenson/GPLFR (and its deps ``pyro-ppl beartype
jaxtyping`` are installed). Runs ``compare_upstream.py`` in a subprocess to keep
the upstream package (also named ``gplfr``) from clashing with ours in-process.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

UP_DIR = os.environ.get("GPLFR_UPSTREAM_DIR")


requires_upstream = pytest.mark.skipif(
    not (UP_DIR and Path(UP_DIR).exists()),
    reason="set GPLFR_UPSTREAM_DIR to a GPLFR clone to run the upstream checks",
)


def _run(script_name):
    script = Path(__file__).resolve().parent / script_name
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, env={**os.environ, "GPLFR_UPSTREAM_DIR": UP_DIR},
    )


@requires_upstream
def test_core_matches_upstream():
    """Exact: the ported collapsed-decoder math equals upstream on identical inputs."""
    result = _run("compare_upstream.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout, result.stdout


@requires_upstream
def test_fit_agrees_with_upstream():
    """Behavioral: both models, fit on the same synthetic data, learn the signal
    and agree on the predicted structure (not an exact match — different GPs)."""
    result = _run("compare_upstream_fit.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout, result.stdout
