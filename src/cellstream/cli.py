"""`cellstream` command-line interface.

Subcommands: `info` (packed member table) and `runtime` (Rust acceleration
status). `convert`, `pack` and `unpack` operated on the non-packed v1 /
directory archives and were removed with them in #154 — and with them the
`PackedOnlyError` -> `SystemExit` wrapper (ultrareview L9), since neither
remaining subcommand can reach the non-packed guard.
"""

from __future__ import annotations

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="cellstream", description="cellstream CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    nf = sub.add_parser("info", help="print the member table of a packed file")
    nf.add_argument("file", help="path to the packed file")

    sub.add_parser(
        "runtime",
        help="print one line describing the Rust acceleration runtime (#244)",
    )
    args = p.parse_args(argv)

    if args.cmd == "runtime":
        from cellstream.runtime import rust_status

        st = rust_status()
        # One greppable line so a benchmark log records which path actually ran.
        detail = st["reason"] or "extension present and enabled"
        print(f"cellstream {st['version']} | rust={'yes' if st['available'] else 'no'} | {detail}")
        return 0

    if args.cmd == "info":
        from cellstream.packed.reader import PackedArchive

        pa = PackedArchive(args.file)
        total = max((m["offset"] + m["length"]) for m in pa.members)
        print(f"packed archive: {args.file}")
        print(
            f"  container_version={pa.footer['container_version']} "
            f"alignment={pa.footer['alignment']}"
        )
        print(
            f"  shard_region_end={pa.shard_region_end} members={len(pa.members)} body_end={total}"
        )
        for m in pa.members:
            idx = f" idx={m['shard_idx']}" if "shard_idx" in m else ""
            print(
                f"  [{m['role']:>8}] {m['name']:<20} offset={m['offset']:<14} "
                f"length={m['length']}{idx}"
            )
        return 0

    raise SystemExit(f"unknown subcommand: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
