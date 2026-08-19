# tests/test_workers_omp_env.py
"""Unit tests for workers.omp_nthreads_in_env.

Part A restored a sibling block here for ``blosc_nthreads_in_env``: its only coverage had
lived in tests/test_read_sharded.py, deleted with the v1 READ surface, while the context
manager itself still had callers. #154 part B removes the v1 WRITER, which was the last of
them -- ``_write_sharded_from_path``'s spawn pool -- so the function is gone from
cellstream/workers.py and those three tests go with it. OMP_NUM_THREADS is the one env window
left, and both surviving worker families (the v2 writer and the v2 reader) use it.
"""

from __future__ import annotations

import os

from cellstream.workers import omp_nthreads_in_env


def test_sets_and_restores_when_unset():
    prev = os.environ.pop("OMP_NUM_THREADS", None)
    try:
        with omp_nthreads_in_env(4):
            assert os.environ["OMP_NUM_THREADS"] == "4"
        assert "OMP_NUM_THREADS" not in os.environ
    finally:
        if prev is not None:
            os.environ["OMP_NUM_THREADS"] = prev


def test_restores_prior_value():
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "9"
        with omp_nthreads_in_env(4):
            assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["OMP_NUM_THREADS"] == "9"
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


def test_none_is_noop():
    prev = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "9"
        with omp_nthreads_in_env(None):
            assert os.environ["OMP_NUM_THREADS"] == "9"
        assert os.environ["OMP_NUM_THREADS"] == "9"
    finally:
        if prev is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = prev


# The blosc_nthreads_in_env block that used to follow (sets-and-restores, restores-prior,
# None-is-noop) is gone with the context manager itself -- see the module docstring.
