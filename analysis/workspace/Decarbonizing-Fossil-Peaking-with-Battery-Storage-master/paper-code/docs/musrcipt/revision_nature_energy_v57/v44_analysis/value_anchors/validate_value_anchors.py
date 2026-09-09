from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def main() -> None:
    cap = pd.read_csv(ROOT / "capacity_market_anchors.csv")
    voll = pd.read_csv(ROOT / "voll_anchors.csv")
    scarcity = pd.read_csv(ROOT / "scarcity_price_anchors.csv")
    scenarios = pd.read_csv(ROOT / "recommended_value_scenarios.csv")
    equivalence = pd.read_csv(ROOT / "vf_voll_equivalence.csv")

    local = cap[cap["reported_unit"].isin(["GBP/kW/year", "PLN/kW/year"])]
    expected_local = (
        local["reported_value"] / local["ecb_local_currency_per_eur"]
    )
    np.testing.assert_allclose(
        expected_local,
        local["converted_eur_kw_year"],
        rtol=0,
        atol=5e-5,
    )

    euro = cap[cap["reported_unit"] == "EUR/MW/year"]
    np.testing.assert_allclose(
        euro["reported_value"] / 1000,
        euro["converted_eur_kw_year"],
        rtol=0,
        atol=5e-5,
    )

    for vf, column in [
        (103.8, "required_avoided_unserved_hours_per_kw_for_vf_103p8"),
        (575.0, "required_avoided_unserved_hours_per_kw_for_vf_575"),
    ]:
        expected_hours = vf * 1000 / equivalence["voll_eur_mwh"]
        np.testing.assert_allclose(
            expected_hours,
            equivalence[column],
            rtol=0,
            atol=0.006,
        )

    priority = cap[cap["jurisdiction"].isin(
        ["Great Britain", "Belgium", "Italy", "Poland"]
    )]
    lo = priority["converted_eur_kw_year"].min()
    hi = priority["converted_eur_kw_year"].max()
    assert np.isclose(lo, 15.6946)
    assert np.isclose(hi, 90.4316)
    assert voll["voll_eur_mwh"].min() == 7000
    assert voll["voll_eur_mwh"].max() == 69000
    assert {350, 1000, 4000}.issubset(set(scarcity["value_eur_mwh"]))
    assert set(scenarios["scenario_family"]) == {
        "capacity_market",
        "voll_eens",
        "scarcity_adder",
    }

    print(
        "PASS | capacity rows=", len(cap),
        "| VOLL rows=", len(voll),
        "| scarcity rows=", len(scarcity),
        "| priority range=", f"{lo:.4f}-{hi:.4f}",
    )


if __name__ == "__main__":
    main()
