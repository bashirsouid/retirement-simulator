"""Unit tests for monte_carlo_portfolio — ledger, data, and config contracts."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from statistics import fmean

import monte_carlo_portfolio as sim

DEFAULT_BLOCK_SIZES = (5, 10, 15, 20)


def make_cfg(**overrides):
    data = dict(
        starting_age=38,
        retirement_age=65,
        end_age=85,
        starting_wealth=500_000.0,
        annual_spending=40_000.0,
        monthly_contribution=2_000.0,
        spending_volatility=0.0,
        charity_pct=0.0,
        equity_allocation=1.0,
        social_security_start_age=999,
        social_security_annual=0.0,
        simulations=10,
        cash_wedge_years=0.0,
        cash_wedge_refill_rule="five_year_mean",
        cash_wedge_escape_velocity=0.25,
        cash_return_override=None,
        bond_return_override=None,
        target_equity_cagr=0.07,
        target_inflation=0.03,
        one_time_events={},
        bootstrap_block_sizes=DEFAULT_BLOCK_SIZES,
        stress_first_n_years=0,
    )
    data.update(overrides)
    return sim.SimulationConfig(**data)


@contextmanager
def override_bootstrap(equity, inflation):
    """Replace the bootstrap with an exact injected path."""
    eq = list(equity)
    inf = list(inflation)

    def _fake(cfg, rng, force_bad_opening=False):
        horizon = cfg.end_age - cfg.starting_age + 1
        if len(eq) != horizon or len(inf) != horizon:
            raise AssertionError(
                f"fixture length eq={len(eq)} inf={len(inf)} != horizon {horizon}"
            )
        return list(eq), list(inf)

    old = sim.bootstrap_historical_series
    sim.bootstrap_historical_series = _fake
    try:
        yield
    finally:
        sim.bootstrap_historical_series = old


def _geo_mean(values):
    return math.exp(fmean(math.log1p(x) for x in values)) - 1.0


def _invert_anchored(series, target_cagr, global_log_mean):
    target_log = math.log1p(target_cagr)
    return [
        math.exp(math.log1p(v) - target_log + global_log_mean) - 1.0
        for v in series
    ]


def _minimal_toml(**overrides):
    data = {
        "starting_age": 38,
        "retirement_age": 65,
        "end_age": 100,
        "starting_wealth": 1_000_000,
        "annual_spending": 40_000,
        "monthly_contribution": 0,
        "equity_allocation": 1.0,
        "simulations": 1,
    }
    data.update(overrides)
    lines = [
        f"{key} = {value!r}" if isinstance(value, str) else f"{key} = {value}"
        for key, value in data.items()
    ]
    return "\n".join(lines) + "\n"


class HistoricalDataTests(unittest.TestCase):

    def test_series_are_aligned_1970_to_2023(self):
        self.assertEqual(len(sim.MSCI_WORLD_ANNUAL), 54)
        self.assertEqual(len(sim.CPI_ANNUAL), 54)
        self.assertEqual(len(sim.MSCI_WORLD_ANNUAL), len(sim.CPI_ANNUAL))

    def test_cpi_contains_known_bls_annual_averages(self):
        by_year = dict(zip(range(1970, 2024), sim.CPI_ANNUAL))
        self.assertAlmostEqual(by_year[1974], 0.124, places=6)
        self.assertAlmostEqual(by_year[1979], 0.113, places=6)
        self.assertAlmostEqual(by_year[1980], 0.135, places=6)
        self.assertAlmostEqual(by_year[2008], 0.038, places=6)
        self.assertAlmostEqual(by_year[2009], -0.004, places=6)
        self.assertAlmostEqual(by_year[2021], 0.047, places=6)
        self.assertAlmostEqual(by_year[2022], 0.080, places=6)
        self.assertAlmostEqual(by_year[2023], 0.041, places=6)

    def test_msci_contains_known_calendar_years(self):
        by_year = dict(zip(range(1970, 2024), sim.MSCI_WORLD_ANNUAL))
        self.assertAlmostEqual(by_year[2008], -0.4033, places=6)
        self.assertAlmostEqual(by_year[2009], 0.3079, places=6)
        self.assertAlmostEqual(by_year[2022], -0.1773, places=6)

    def test_2008_equity_crash_is_paired_with_2008_cpi(self):
        year_2008 = 2008 - 1970
        self.assertAlmostEqual(sim.MSCI_WORLD_ANNUAL[year_2008], -0.4033, places=6)
        self.assertAlmostEqual(sim.CPI_ANNUAL[year_2008], 0.038, places=6)


class ShiftTests(unittest.TestCase):

    def test_empty_series_is_returned_unchanged(self):
        self.assertEqual(sim._shift_to_target_anchored([], 0.07, 0.09), [])

    def test_uses_global_mean_not_path_mean(self):
        out = sim._shift_to_target_anchored([-0.4033], 0.07, sim.MSCI_WORLD_GEO_LOG_MEAN)
        expected = math.exp(
            math.log1p(-0.4033) - sim.MSCI_WORLD_GEO_LOG_MEAN + math.log1p(0.07)
        ) - 1.0
        self.assertAlmostEqual(out[0], expected, places=12)

    def test_path_geometric_mean_is_not_forced_to_target(self):
        out = sim._shift_to_target_anchored(
            [-0.4033, -0.1954, -0.1773],
            0.07,
            sim.MSCI_WORLD_GEO_LOG_MEAN,
        )
        self.assertLess(_geo_mean(out), 0.0)

    def test_floor_prevents_total_loss_below_minus_one(self):
        out = sim._shift_to_target_anchored([-0.999999], -0.5, 0.09)
        self.assertGreaterEqual(out[0], -0.9999)


class BootstrapTests(unittest.TestCase):

    def test_series_length_matches_horizon(self):
        for end_age in (60, 75, 100):
            cfg = make_cfg(end_age=end_age)
            eq, inf = sim.bootstrap_historical_series(cfg, random.Random(0))
            horizon = cfg.end_age - cfg.starting_age + 1
            self.assertEqual(len(eq), horizon)
            self.assertEqual(len(inf), horizon)

    def test_values_come_from_historical_arrays(self):
        cfg = make_cfg(end_age=80)
        eq, inf = sim.bootstrap_historical_series(cfg, random.Random(42))
        for v in eq:
            self.assertGreater(v, -1.0)
            self.assertLess(v, 2.0)
        for v in inf:
            self.assertGreater(v, -0.5)
            self.assertLess(v, 0.5)

    def test_bootstrapped_equity_inverts_to_historical_blocks(self):
        cfg = make_cfg(end_age=70, bootstrap_block_sizes=(5,))
        eq, _ = sim.bootstrap_historical_series(cfg, random.Random(3))
        raw = _invert_anchored(eq, cfg.target_equity_cagr, sim.MSCI_WORLD_GEO_LOG_MEAN)
        hist = sim.MSCI_WORLD_ANNUAL
        for offset in range(0, len(raw), 5):
            chunk = raw[offset:offset + 5]
            matched = any(
                all(abs(a - b) < 1e-12 for a, b in zip(chunk, hist[start:start + len(chunk)]))
                for start in range(0, len(hist) - len(chunk) + 1)
            )
            self.assertTrue(matched, f"no historical match for block at offset {offset}")

    def test_deterministic_given_same_seed(self):
        cfg = make_cfg(end_age=70)
        eq_a, inf_a = sim.bootstrap_historical_series(cfg, random.Random(99))
        eq_b, inf_b = sim.bootstrap_historical_series(cfg, random.Random(99))
        self.assertEqual(eq_a, eq_b)
        self.assertEqual(inf_a, inf_b)

    def test_different_seeds_differ(self):
        cfg = make_cfg(end_age=70)
        eq_a, _ = sim.bootstrap_historical_series(cfg, random.Random(1))
        eq_b, _ = sim.bootstrap_historical_series(cfg, random.Random(2))
        self.assertNotEqual(eq_a, eq_b)

    def test_cagr_shift_applied(self):
        cfg = make_cfg(end_age=100, target_equity_cagr=0.07)
        eq, _ = sim.bootstrap_historical_series(cfg, random.Random(7))
        self.assertAlmostEqual(_geo_mean(eq), 0.07, delta=0.015)

    def test_inflation_shift_applied(self):
        cfg = make_cfg(end_age=100, target_inflation=0.03)
        _, inf = sim.bootstrap_historical_series(cfg, random.Random(13))
        self.assertAlmostEqual(_geo_mean(inf), 0.03, delta=0.01)


class ConfigValidationTests(unittest.TestCase):

    def test_default_fixture_is_valid(self):
        sim.validate_config(make_cfg())

    def test_partial_bond_allocation_requires_bond_return(self):
        cfg = make_cfg(equity_allocation=0.80, bond_return_override=None)
        with self.assertRaisesRegex(ValueError, "bond_return_override"):
            sim.validate_config(cfg)

    def test_explicit_bond_return_is_accepted(self):
        cfg = make_cfg(equity_allocation=0.80, bond_return_override=0.03)
        sim.validate_config(cfg)

    def test_end_age_before_start_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "end_age"):
            sim.validate_config(make_cfg(starting_age=80, end_age=70))

    def test_zero_simulations_rejected(self):
        with self.assertRaisesRegex(ValueError, "simulations"):
            sim.validate_config(make_cfg(simulations=0))

    def test_negative_wedge_rejected(self):
        with self.assertRaisesRegex(ValueError, "cash_wedge_years"):
            sim.validate_config(make_cfg(cash_wedge_years=-1.0))

    def test_load_config_validates_non_equity_allocation(self):
        text = _minimal_toml(equity_allocation=0.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "bond_return_override"):
                sim.load_config(str(path))

    def test_load_config_parses_one_time_events_as_int_ages(self):
        text = _minimal_toml() + '[one_time_events]\n"55" = -50000\n"65" = 200000\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(text)
            cfg = sim.load_config(str(path))
        self.assertEqual(cfg.one_time_events, {55: -50_000.0, 65: 200_000.0})

    def test_load_config_ignores_unknown_keys(self):
        text = _minimal_toml() + "avg_invest_return = 0.99\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(text)
            cfg = sim.load_config(str(path))
        self.assertEqual(cfg.target_equity_cagr, sim.DEFAULT_TARGET_EQUITY_CAGR)

    def test_starting_after_retirement_is_allowed(self):
        sim.validate_config(make_cfg(starting_age=70, retirement_age=65, end_age=90))


class SimulatePathTests(unittest.TestCase):

    def test_simulate_path_is_deterministic(self):
        cfg = make_cfg(end_age=70)
        a = sim.simulate_path(cfg, seed=11)
        b = sim.simulate_path(cfg, seed=11)
        self.assertEqual(
            [yr.total_wealth for yr in a.years],
            [yr.total_wealth for yr in b.years],
        )

    def test_first_year_cumulative_inflation_is_one(self):
        path = sim.simulate_path(make_cfg(end_age=40), seed=0)
        self.assertAlmostEqual(path.years[0].cumulative_inflation, 1.0, places=12)

    def test_earlier_retirement_gives_lower_wealth(self):
        cfg_early = make_cfg(retirement_age=40, end_age=60)
        cfg_late = make_cfg(retirement_age=55, end_age=60)
        early = [sim.simulate_path(cfg_early, s).years[-1].total_wealth for s in range(20)]
        late = [sim.simulate_path(cfg_late, s).years[-1].total_wealth for s in range(20)]
        self.assertLess(sum(early) / len(early), sum(late) / len(late))

    def test_base_spending_starts_inflation_adjusted(self):
        cfg = make_cfg(
            starting_age=38,
            retirement_age=43,
            end_age=50,
            annual_spending=40_000.0,
            spending_volatility=0.0,
            target_inflation=0.03,
        )
        path = sim.simulate_path(cfg, seed=0)
        first = next(yr for yr in path.years if yr.age == cfg.retirement_age)
        self.assertGreater(first.base_spending, cfg.annual_spending)

    def test_first_retirement_spending_uses_start_of_year_index(self):
        cfg = make_cfg(
            starting_age=38,
            retirement_age=40,
            end_age=40,
            annual_spending=40_000.0,
            target_inflation=0.03,
        )
        path = sim.simulate_path(cfg, seed=0)
        first = path.years[-1]
        self.assertAlmostEqual(first.base_spending, 40_000.0 * first.cumulative_inflation, places=6)

    def test_base_spending_tracks_realized_inflation_after_retirement(self):
        cfg = make_cfg(
            starting_age=60,
            retirement_age=60,
            end_age=61,
            starting_wealth=1_000_000.0,
            annual_spending=40_000.0,
            monthly_contribution=0.0,
        )
        with override_bootstrap([0.0, 0.0], [0.10, 0.00]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].base_spending, 40_000.0, places=6)
        self.assertAlmostEqual(path.years[1].base_spending, 44_000.0, places=6)
        self.assertAlmostEqual(path.years[1].cumulative_inflation, 1.10, places=6)

    def test_social_security_compounds_with_inflation(self):
        cfg = make_cfg(
            starting_age=38,
            retirement_age=40,
            end_age=80,
            social_security_start_age=45,
            social_security_annual=20_000.0,
            target_inflation=0.03,
        )
        path = sim.simulate_path(cfg, seed=5)
        ss_years = [
            yr for yr in path.years
            if yr.age >= cfg.social_security_start_age and yr.social_security_income > 0
        ]
        self.assertGreater(len(ss_years), 0)
        for i in range(1, len(ss_years)):
            self.assertGreaterEqual(
                ss_years[i].social_security_income,
                ss_years[i - 1].social_security_income * 0.99,
            )

    def test_ss_uses_start_of_year_inflation_index(self):
        cfg = make_cfg(
            starting_age=62,
            retirement_age=62,
            end_age=63,
            starting_wealth=1_000_000.0,
            annual_spending=10_000.0,
            monthly_contribution=0.0,
            social_security_start_age=62,
            social_security_annual=30_000.0,
        )
        with override_bootstrap([0.0, 0.0], [0.10, 0.00]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].social_security_income, 30_000.0, places=6)
        self.assertAlmostEqual(path.years[1].social_security_income, 33_000.0, places=6)

    def test_ss_already_claimed_when_starting_after_claim_age(self):
        cfg = make_cfg(
            starting_age=67,
            retirement_age=67,
            end_age=67,
            social_security_start_age=62,
            social_security_annual=30_000.0,
        )
        path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].social_security_income, 30_000.0, places=6)

    def test_exact_zero_assets_count_as_depletion(self):
        cfg = make_cfg(
            starting_age=38,
            retirement_age=38,
            end_age=38,
            starting_wealth=100_000.0,
            annual_spending=100_000.0,
            monthly_contribution=0.0,
        )
        with override_bootstrap([0.0], [0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertTrue(path.depleted_any_time)
        self.assertTrue(path.terminal_depleted)
        self.assertEqual(path.years[0].total_wealth, 0.0)

    def test_one_time_event_is_inflation_indexed(self):
        cfg = make_cfg(
            starting_age=38,
            retirement_age=100,
            end_age=39,
            starting_wealth=0.0,
            monthly_contribution=0.0,
            one_time_events={39: 100_000.0},
        )
        with override_bootstrap([0.0, 0.0], [0.10, 0.10]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[-1].total_wealth, 110_000.0, places=6)

    def test_contribution_does_not_receive_prior_year_return(self):
        cfg = make_cfg(
            starting_age=38,
            retirement_age=40,
            end_age=38,
            starting_wealth=0.0,
            monthly_contribution=100.0,
        )
        with override_bootstrap([0.10], [0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].total_wealth, 1_200.0, places=6)
        self.assertAlmostEqual(path.years[0].contribution, 1_200.0, places=6)

    def test_contribution_stops_at_retirement_age(self):
        cfg = make_cfg(
            starting_age=64,
            retirement_age=65,
            end_age=65,
            monthly_contribution=100.0,
        )
        with override_bootstrap([0.0, 0.0], [0.0, 0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].contribution, 1_200.0, places=6)
        self.assertAlmostEqual(path.years[1].contribution, 0.0, places=6)

    def test_zero_inflation_two_year_ledger(self):
        cfg = make_cfg(
            starting_age=60,
            retirement_age=60,
            end_age=61,
            starting_wealth=100_000.0,
            annual_spending=10_000.0,
            monthly_contribution=0.0,
        )
        with override_bootstrap([0.10, 0.00], [0.0, 0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].total_wealth, 90_000.0, places=6)
        self.assertAlmostEqual(path.years[1].total_wealth, 89_000.0, places=6)

    def test_bond_sleeve_return_is_visible_on_next_opening_balance(self):
        cfg = make_cfg(
            starting_age=40,
            retirement_age=50,
            end_age=41,
            starting_wealth=100_000.0,
            monthly_contribution=0.0,
            equity_allocation=0.0,
            bond_return_override=0.04,
        )
        with override_bootstrap([0.50, 0.50], [0.0, 0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].total_wealth, 100_000.0, places=6)
        self.assertAlmostEqual(path.years[1].total_wealth, 104_000.0, places=6)

    def test_depletion_flag_set_on_high_spend(self):
        cfg = make_cfg(
            retirement_age=38,
            end_age=50,
            starting_wealth=100_000.0,
            annual_spending=200_000.0,
            monthly_contribution=0.0,
            cash_wedge_years=0.0,
        )
        self.assertTrue(any(sim.simulate_path(cfg, s).depleted_any_time for s in range(50)))

    def test_no_depletion_on_very_low_spend(self):
        cfg = make_cfg(
            retirement_age=38,
            end_age=80,
            starting_wealth=10_000_000.0,
            annual_spending=1_000.0,
            monthly_contribution=0.0,
        )
        self.assertFalse(any(sim.simulate_path(cfg, s).depleted_any_time for s in range(100)))

    def test_one_time_event_inflow_increases_wealth(self):
        cfg_base = make_cfg(retirement_age=40, end_age=60, one_time_events={})
        cfg_event = make_cfg(retirement_age=40, end_age=60, one_time_events={50: 1_000_000.0})
        base = [sim.simulate_path(cfg_base, s).years[-1].total_wealth for s in range(20)]
        event = [sim.simulate_path(cfg_event, s).years[-1].total_wealth for s in range(20)]
        self.assertGreater(sum(event) / len(event), sum(base) / len(base))

    def test_charity_reduces_wealth(self):
        cfg_no = make_cfg(retirement_age=40, end_age=70, charity_pct=0.0)
        cfg_yes = make_cfg(retirement_age=40, end_age=70, charity_pct=0.05)
        no = [sim.simulate_path(cfg_no, s).years[-1].total_wealth for s in range(20)]
        yes = [sim.simulate_path(cfg_yes, s).years[-1].total_wealth for s in range(20)]
        self.assertGreater(sum(no) / len(no), sum(yes) / len(yes))


class WedgeTests(unittest.TestCase):

    def _wedge_cfg(self, **overrides):
        data = dict(
            starting_age=60,
            retirement_age=60,
            end_age=60,
            starting_wealth=200_000.0,
            annual_spending=40_000.0,
            monthly_contribution=0.0,
            cash_wedge_years=1.0,
            cash_wedge_escape_velocity=0.0,
            cash_wedge_refill_rule="negative_year",
        )
        data.update(overrides)
        return make_cfg(**data)

    def test_wedge_is_funded_and_kept_in_an_up_year(self):
        with override_bootstrap([0.10], [0.0]):
            path = sim.simulate_path(self._wedge_cfg(), seed=0)
        self.assertAlmostEqual(path.years[0].wedge_cash, 40_000.0, places=6)
        self.assertAlmostEqual(path.years[0].portfolio_wealth, 120_000.0, places=6)
        self.assertAlmostEqual(path.years[0].total_wealth, 160_000.0, places=6)

    def test_down_year_spends_from_wedge_not_portfolio(self):
        with override_bootstrap([-0.20], [0.0]):
            path = sim.simulate_path(self._wedge_cfg(), seed=0)
        self.assertAlmostEqual(path.years[0].portfolio_wealth, 160_000.0, places=6)
        self.assertAlmostEqual(path.years[0].wedge_cash, 0.0, places=6)

    def test_wedge_funds_when_simulation_starts_already_retired(self):
        cfg = self._wedge_cfg(starting_age=70, retirement_age=65, end_age=70)
        with override_bootstrap([0.10], [0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].wedge_cash, 40_000.0, places=6)
        self.assertAlmostEqual(path.years[0].portfolio_wealth, 120_000.0, places=6)

    def test_ss_surplus_can_retire_the_wedge(self):
        cfg = make_cfg(
            starting_age=60,
            retirement_age=60,
            end_age=60,
            starting_wealth=500_000.0,
            annual_spending=40_000.0,
            monthly_contribution=0.0,
            cash_wedge_years=1.0,
            cash_wedge_escape_velocity=0.25,
            social_security_start_age=60,
            social_security_annual=50_000.0,
        )
        with override_bootstrap([0.0], [0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].wedge_cash, 0.0, places=6)
        self.assertAlmostEqual(path.years[0].total_wealth, 510_000.0, places=6)
        self.assertFalse(path.wedge_depleted)


class GuardrailTests(unittest.TestCase):

    def test_guardrail_cuts_base_spending_after_a_crash(self):
        cfg = make_cfg(
            starting_age=60,
            retirement_age=60,
            end_age=61,
            starting_wealth=100_000.0,
            annual_spending=20_000.0,
            monthly_contribution=0.0,
        )
        with override_bootstrap([-0.50, 0.0], [0.0, 0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].base_spending, 20_000.0, places=6)
        self.assertAlmostEqual(path.years[1].base_spending, 18_000.0, places=6)
        self.assertAlmostEqual(path.years[1].actual_spending, 18_000.0, places=6)

    def test_healthy_path_does_not_raise_spending(self):
        cfg = make_cfg(
            starting_age=60,
            retirement_age=60,
            end_age=61,
            starting_wealth=100_000.0,
            annual_spending=20_000.0,
            monthly_contribution=0.0,
        )
        with override_bootstrap([0.50, 0.50], [0.0, 0.0]):
            path = sim.simulate_path(cfg, seed=0)
        self.assertAlmostEqual(path.years[0].base_spending, 20_000.0, places=6)
        self.assertAlmostEqual(path.years[1].base_spending, 20_000.0, places=6)


class StressTests(unittest.TestCase):

    def test_stress_keeps_pre_retirement_and_tail_blocks(self):
        cfg = make_cfg(starting_age=40, retirement_age=50, end_age=70, stress_first_n_years=5)
        eq_n, inf_n = sim.bootstrap_historical_series(cfg, random.Random(0), False)
        eq_s, inf_s = sim.bootstrap_historical_series(cfg, random.Random(0), True)
        pre = 10
        self.assertEqual(eq_n[:pre], eq_s[:pre])
        self.assertEqual(inf_n[:pre], inf_s[:pre])
        self.assertEqual(eq_n[pre + 5:], eq_s[pre + 5:])
        self.assertEqual(inf_n[pre + 5:], inf_s[pre + 5:])

    def test_stress_window_is_a_worst_historical_block(self):
        cfg = make_cfg(starting_age=40, retirement_age=50, end_age=70, stress_first_n_years=5)
        eq_s, inf_s = sim.bootstrap_historical_series(cfg, random.Random(1), True)
        pre = 10
        raw_eq = _invert_anchored(eq_s[pre:pre + 5], cfg.target_equity_cagr, sim.MSCI_WORLD_GEO_LOG_MEAN)
        raw_inf = _invert_anchored(inf_s[pre:pre + 5], cfg.target_inflation, sim.CPI_GEO_LOG_MEAN)
        matched = False
        for start in sim._get_worst_starts(5):
            hist_eq = sim.MSCI_WORLD_ANNUAL[start:start + 5]
            hist_inf = sim.CPI_ANNUAL[start:start + 5]
            if all(abs(a - b) < 1e-9 for a, b in zip(raw_eq, hist_eq)) and all(
                abs(a - b) < 1e-9 for a, b in zip(raw_inf, hist_inf)
            ):
                matched = True
                break
        self.assertTrue(matched, "stressed opening window is not a worst-start block")

    def test_stress_path_is_deterministic(self):
        cfg = make_cfg(stress_first_n_years=10, end_age=80)
        a = sim.bootstrap_historical_series(cfg, random.Random(4), True)
        b = sim.bootstrap_historical_series(cfg, random.Random(4), True)
        self.assertEqual(a, b)


class RunSimulationTests(unittest.TestCase):

    def test_percentiles_monotonically_ordered(self):
        cfg = make_cfg(simulations=200)
        ages, matrix, pct_rows, *_ = sim.run_simulation(cfg)
        for age in ages:
            row = pct_rows[age]
            self.assertLessEqual(row["p01"], row["p25"])
            self.assertLessEqual(row["p25"], row["median"])
            self.assertLessEqual(row["median"], row["p75"])
            self.assertLessEqual(row["p75"], row["p99"])

    def test_matrix_shape(self):
        cfg = make_cfg(simulations=50)
        ages, matrix, *_ = sim.run_simulation(cfg)
        horizon = cfg.end_age - cfg.starting_age + 1
        self.assertEqual(len(ages), horizon)
        self.assertEqual(len(matrix), 50)
        for row in matrix:
            self.assertEqual(len(row), horizon)

    def test_lower_cagr_gives_lower_median(self):
        cfg_hi = make_cfg(simulations=500, target_equity_cagr=0.09, end_age=80)
        cfg_lo = make_cfg(simulations=500, target_equity_cagr=0.04, end_age=80)
        _, _, pct_hi, *_ = sim.run_simulation(cfg_hi)
        _, _, pct_lo, *_ = sim.run_simulation(cfg_lo)
        self.assertGreater(pct_hi[80]["median"], pct_lo[80]["median"])

    def test_run_simulation_return_arity(self):
        result = sim.run_simulation(make_cfg(simulations=5, end_age=50))
        self.assertEqual(len(result), 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
