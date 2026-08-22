"""Build shim so the Rust extension can be made optional CONDITIONALLY (#244).

setuptools-rust reads a STATIC `optional` from pyproject.toml, so `CELLSTREAM_NO_RUST=1`
cannot be honoured there. The declaration therefore lives here and NOT in pyproject.toml --
keeping both would register the extension twice and build it twice.

Default is strict: a source build on a machine with no Rust toolchain FAILS LOUDLY instead of
quietly producing an install whose cell reads run ~10x slower with no indication (issue #244).
Set CELLSTREAM_NO_RUST=1 for a deliberate slim/pure-Python install.

NOTE `optional` is the WRONG knob for that opt-out: RustExtension(optional=True) still ATTEMPTS
the build and merely swallows a failure, so on a node that HAS a toolchain it would still
produce a Rust-enabled install. The extension list itself has to be empty.
"""

import os

from setuptools import setup
from setuptools_rust import Binding, RustExtension

_NO_RUST = os.environ.get("CELLSTREAM_NO_RUST") == "1"

setup(
    rust_extensions=[]
    if _NO_RUST
    else [
        RustExtension(
            "cellstream.v2._rust",
            path="rust/Cargo.toml",
            binding=Binding.PyO3,
            optional=False,
            # Always build the extension in release (opt-level 3 + thin LTO). setuptools-rust
            # otherwise builds editable installs (`pip install -e .`) in DEBUG, which is ~7x
            # slower per thread on the encode hot path — the cause of the spurious "Rust
            # 0.041 GB/s vs Python 0.30" benchmark (a debug `.so`; release ~0.44 GB/s/thread,
            # above Python). Wheels / `pip install .` were already release; this makes dev and
            # bench installs match.
            debug=False,
        )
    ]
)
