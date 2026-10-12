"""Scoring rules on small hand-made metric tables: percentiles, missing data, outliers,
weights, bottleneck labels, filtering and comparison."""

import numpy as np
import pandas as pd
import pytest

from airport_agent import scoring


def make_metrics(n: int = 4, **columns) -> pd.DataFrame:
    """n fake airports A, B, C, ... where every metric rises from A to the last airport.

    Keyword arguments replace whole metric columns.
    """
    codes = [chr(ord("A") + i) for i in range(n)]
    table = pd.DataFrame(
        {metric: np.arange(1, n + 1, dtype=float) for metric in scoring.METRICS},
        index=pd.Index(codes, name="airport"),
    )
    table["name"] = [f"Airport {c}" for c in codes]
    table["state"] = ["MA", "MA", "CA", "CA", "CA", "CA"][:n]
    table["division"] = ["New England", "New England", "Pacific", "Pacific", "Pacific", "Pacific"][:n]
    table["region"] = ["Northeast", "Northeast", "West", "West", "West", "West"][:n]
    table["passengers"] = [1e5, 1e6, 1e7, 2e7, 3e7, 4e7][:n]
    table["small_base"] = False
    for name, values in columns.items():
        table[name] = values
    return table


# --- percentiles --------------------------------------------------------------

def test_percentile_ranks_lowest_zero_highest_hundred():
    assert scoring.percentile_ranks(pd.Series([10.0, 20.0, 30.0])).tolist() == [0.0, 50.0, 100.0]


def test_percentile_ranks_ties_and_missing():
    pct = scoring.percentile_ranks(pd.Series([1.0, 1.0, 2.0, np.nan]))
    assert pct[:3].tolist() == [25.0, 25.0, 100.0]  # ties share the average rank (1.5)
    assert np.isnan(pct[3])                         # missing stays missing


def test_percentile_ranks_single_value_is_fifty():
    assert scoring.percentile_ranks(pd.Series([7.0, np.nan]))[0] == 50.0


# --- missing data -------------------------------------------------------------

def test_missing_fare_scores_unmet_demand_on_load_factor_alone():
    scored = scoring.score(make_metrics(fare_premium=[np.nan, 1.0, 2.0, 3.0]))
    assert scored.loc["A", "unmet_demand"] == scored.loc["A", "load_factor_pct"]
    assert "fare_premium" in scored.loc["A", "missing_metrics"]
    assert any("fare premium" in c for c in scored.loc["A", "caveats"])


def test_missing_group_rescales_weights():
    scored = scoring.score(make_metrics(nas_delay_rate=[np.nan, 1.0, 2.0, 3.0]))
    a = scored.loc["A"]
    expected = (30 * a["runway_pressure"] + 30 * a["terminal_pressure"] + 30 * a["unmet_demand"]) / 90
    assert a["total_score"] == pytest.approx(expected)


def test_no_data_in_any_weighted_group_gives_no_score_and_a_caveat():
    congestion_only = {group: 0 for group in scoring.GROUPS} | {"congestion": 1}
    scored = scoring.score(make_metrics(nas_delay_rate=[np.nan, 1.0, 2.0, 3.0]), weights=congestion_only)
    assert pd.isna(scored.loc["A", "total_score"]) and pd.isna(scored.loc["A", "national_rank"])
    assert scoring.NO_SCORE_CAVEAT in scored.loc["A", "caveats"]
    assert scoring.NO_SCORE_CAVEAT not in scored.loc["B", "caveats"]


# --- outliers -----------------------------------------------------------------

def test_extreme_outlier_does_not_move_other_airports_percentiles():
    normal = scoring.score(make_metrics(passenger_growth=[0.01, 0.02, 0.03, 0.04]))
    extreme = scoring.score(make_metrics(passenger_growth=[0.01, 0.02, 0.03, 5.00]))
    assert extreme.loc["D", "passenger_growth_pct"] == 100.0
    pd.testing.assert_series_equal(
        normal.loc[["A", "B", "C"], "passenger_growth_pct"],
        extreme.loc[["A", "B", "C"], "passenger_growth_pct"],
    )


def test_small_base_gets_its_caveat():
    scored = scoring.score(make_metrics(small_base=[False, False, False, True]))
    assert scoring.SMALL_BASE_CAVEAT in scored.loc["D", "caveats"]
    assert scoring.SMALL_BASE_CAVEAT not in scored.loc["A", "caveats"]


# --- weights ------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"congestion": -1}, {g: 0 for g in scoring.GROUPS}, {"size": 10}])
def test_invalid_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        scoring.score(make_metrics(), weights=bad)


def test_zero_congestion_weight_leaves_the_other_three_groups():
    scored = scoring.score(make_metrics(nas_delay_rate=[4.0, 3.0, 2.0, 1.0]), weights={"congestion": 0})
    b = scored.loc["B"]
    assert b["total_score"] == pytest.approx((b["runway_pressure"] + b["terminal_pressure"] + b["unmet_demand"]) / 3)


def test_runway_fullness_counts_twice_growth():
    scored = scoring.score(make_metrics(flight_growth=[4.0, 3.0, 2.0, 1.0]))  # growth falls as fullness rises
    # A: fullness pct 0, growth pct 100 -> (2*0 + 1*100) / 3
    assert scored.loc["A", "runway_pressure"] == pytest.approx(100 / 3)


# --- bottleneck label ---------------------------------------------------------

def test_bottleneck_labels():
    pct = pd.DataFrame(
        {  # load factor, flights per runway, aviation-system delays (percentiles)
            "load_factor_pct":        [90, 90, 90, 40, np.nan],
            "flights_per_runway_pct": [80, 20, 20, 90, 90],
            "nas_delay_rate_pct":     [10, 95, 20, 90, 90],
        },
        index=["full_runways", "full_by_delays", "room", "planes_not_full", "no_data"],
    )
    labels = scoring.bottleneck_label(pct)
    assert labels.to_dict() == {
        "full_runways": "airport-constrained",
        "full_by_delays": "airline-side",  # delays alone don't make an airport constrained
        "room": "airline-side",
        "planes_not_full": "no unmet-demand signal",
        "no_data": "unknown",
    }


def test_strong_candidate_needs_high_score_and_airport_constraint():
    # 6 airports; every metric rises A -> F, so F has the top score and is full in every way
    scored = scoring.score(make_metrics(n=6))
    assert bool(scored.loc["F", "strong_candidate"]) is True     # top score, airport-constrained
    assert bool(scored.loc["A", "strong_candidate"]) is False    # lowest score, planes not full
    # high score but the airport has room (few flights per runway) -> not a strong candidate
    roomy = scoring.score(make_metrics(n=6, flights_per_runway=[6.0, 5, 4, 3, 2, 1], nas_delay_rate=[6.0, 5, 4, 3, 2, 1]))
    assert roomy.loc["F", "bottleneck"] == "airline-side"
    assert bool(roomy.loc["F", "strong_candidate"]) is False


# --- ranking and filtering ----------------------------------------------------

def test_region_filter_does_not_change_scores():
    scored = scoring.score(make_metrics())
    new_england = scoring.rank_airports(scored, area="new england")  # case-insensitive
    assert list(new_england.index) == ["B", "A"]
    for code in new_england.index:
        assert new_england.loc[code, "total_score"] == scored.loc[code, "total_score"]
    assert new_england["regional_rank"].tolist() == [1, 2]
    assert new_england["regional_count"].tolist() == [2, 2]


def test_size_and_state_filters():
    scored = scoring.score(make_metrics())
    assert list(scoring.rank_airports(scored, min_passengers=1e7, max_passengers=2e7).index) == ["D", "C"]
    assert list(scoring.rank_airports(scored, state="ca", top_n=1).index) == ["D"]


def test_unknown_area_is_rejected():
    with pytest.raises(ValueError, match="Census division or region"):
        scoring.rank_airports(scoring.score(make_metrics()), area="Atlantis")


# --- comparison ---------------------------------------------------------------

def test_compare_airports_computes_differences():
    scored = scoring.score(make_metrics())
    table = scoring.compare_airports(scored, ["A", "D"])
    assert table.loc["load_factor_pct", "A"] == 0.0
    assert table.loc["load_factor_pct", "D"] == 100.0
    assert table.loc["load_factor_pct", "D - A"] == 100.0


def test_compare_unknown_airport_is_rejected():
    with pytest.raises(KeyError):
        scoring.compare_airports(scoring.score(make_metrics()), ["A", "ZZZ"])
