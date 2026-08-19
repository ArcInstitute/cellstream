from __future__ import annotations

import pandas as pd

from cellstream import header


def test_var_parquet_round_trip(tmp_path):
    var = pd.DataFrame({"gene": ["A", "B"]}, index=["g0", "g1"])
    p = tmp_path / "var" / "raw.parquet"
    p.parent.mkdir()
    header.write_var_parquet(var, p)
    back = header.read_var_parquet(p)
    assert list(back.index) == ["g0", "g1"]
    assert list(back["gene"]) == ["A", "B"]
