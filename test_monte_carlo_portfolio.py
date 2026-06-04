"""Unit tests for monte_carlo_portfolio — global equity block bootstrap engine."""

import math
import random
import sys
import unittest
from statistics import fmean

import monte_carlo_portfolio as sim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZES = (5, 10, 15, 20)


def make_cfg(**overrides):
    """Return a minimal but valid SimulationConfig, with any field overridable."""
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
    )
    data.update(overrides)
    return sim.SimulationConfig(**data)


# ---------------------------------------------------------------------------
# BootstrapTests
# ---------------------------------------------------------------------------

class BootstrapTests(unittest.TestCase):

    def test_series_length_matches_horizon(self):
        for end_age in (60, 75, 100):
            cfg = make_cfg(end_age=end_age)
            rng = random.Random(0)
            eq, inf = sim.bootstrap_historical_series(cfg, rng)
            horizon = cfg.end_age - cfg.starting_age + 1
            self.assertEqual(len(eq), horizon,
                             f"eq length {len(eq)} != horizon {horizon} for end_age={end_age}")
            self.assertEqual(len(inf), horizon,
                             f"inf length {len(inf)} != horizon {horizon} for end_age={end_age}")

    def test_values_come_from_historical_arrays(self):
        """After CAGR shifting, values must still be within a plausible range."""
        cfg = make_cfg(end_age=80)
        rng = random.Random(42)
        eq, inf = sim.bootstrap_historical_series(cfg, rng)
        for v in eq:
            self.assertGreater(v, -1.0, "equity return below -100%")
            self.assertLess(v, 2.0,    "equity return implausibly above +200%")
        for v in inf:
            self.assertGreater(v, -0.5, "inflation below -50%")
            self.assertLess(v, 0.5,    "inflation above +50%")

    def test_deterministic_given_same_seed(self):
        cfg = make_cfg(end_age=70)
        rng_a = random.Random(99)
        rng_b = random.Random(99)
        eq_a, inf_a = sim.bootstrap_historical_series(cfg, rng_a)
        eq_b, inf_b = sim.bootstrap_historical_series(cfg, rng_b)
        self.assertEqual(eq_a, eq_b)
        self.assertEqual(inf_a, inf_b)

    def test_different_seeds_differ(self):
        cfg = make_cfg(end_age=70)
        rng_a = random.Random(1)
        rng_b = random.Random(2)
        eq_a, _ = sim.bootstrap_historical_series(cfg, rng_a)
        eq_b, _ = sim.bootstrap_historical_series(cfg, rng_b)
        self.assertNotEqual(eq_a, eq_b, "Different seeds should produce different series")

    def test_cagr_shift_applied(self):
        """Geometric mean of the returned equity series should be close to target_equity_cagr."""
        cfg = make_cfg(end_age=100, target_equity_cagr=0.07)
        rng = random.Random(7)
        eq, _ = sim.bootstrap_historical_series(cfg, rng)
        actual_cagr = math.exp(fmean(math.log1p(x) for x in eq)) - 1.0
        self.assertAlmostEqual(actual_cagr, 0.07, delta=0.001,
                               msg=f"Shifted CAGR {actual_cagr:.4f} not near 0.07")

    def test_inflation_shift_applied(self):
        """Geometric mean of the returned inflation series should be close to target_inflation."""
        cfg = make_cfg(end_age=100, target_inflation=0.03)
        rng = random.Random(13)
        _, inf = sim.bootstrap_historical_series(cfg, rng)
        actual = math.exp(fmean(math.log1p(x) for x in inf)) - 1.0
        self.assertAlmostEqual(actual, 0.03, delta=0.001,
                               msg=f"Shifted inflation CAGR {actual:.4f} not near 0.03")


# ---------------------------------------------------------------------------
# SimulatePathTests
# ---------------------------------------------------------------------------

class SimulatePathTests(unittest.TestCase):

    def test_earlier_retirement_gives_lower_wealth(self):
        """All else equal, retiring earlier means fewer contribution years → lower terminal wealth."""
        cfg_early = make_cfg(retirement_age=40, end_age=60)
        cfg_late  = make_cfg(retirement_age=55, end_age=60)
        early_finals = [sim.simulate_path(cfg_early, s).years[-1].total_wealth for s in range(20)]
        late_finals  = [sim.simulate_path(cfg_late,  s).years[-1].total_wealth for s in range(20)]
        self.assertLess(
            sum(early_finals) / len(early_finals),
            sum(late_finals) / len(late_finals),
            "Earlier retirement should produce lower average terminal wealth",
        )

    def test_base_spending_starts_inflation_adjusted(self):
        """Base spending at retirement should reflect inflation from starting_age onward."""
        cfg = make_cfg(
            starting_age=38,
            retirement_age=43,  # 5 years of inflation
            end_age=50,
            annual_spending=40_000.0,
            spending_volatility=0.0,
            target_inflation=0.03,
        )
        path = sim.simulate_path(cfg, seed=0)
        first_retirement_year = next(yr for yr in path.years if yr.age == cfg.retirement_age)
        # Base spending must be >= the nominal annual_spending (it's been inflated)
        self.assertGreater(
            first_retirement_year.base_spending,
            cfg.annual_spending,
            "Base spending at retirement should exceed today's-dollar annual_spending after inflation",
        )

    def test_social_security_compounds_with_inflation(self):
        """Once SS starts, each year's SS income should be >= the prior year's."""
        cfg = make_cfg(
            starting_age=38,
            retirement_age=40,
            end_age=80,
            social_security_start_age=45,
            social_security_annual=20_000.0,
            target_inflation=0.03,
        )
        path = sim.simulate_path(cfg, seed=5)
        ss_years = [yr for yr in path.years if yr.age >= cfg.social_security_start_age
                    and yr.social_security_income > 0]
        self.assertGreater(len(ss_years), 0, "Expected some SS income years")
        for i in range(1, len(ss_years)):
            prev = ss_years[i - 1].social_security_income
            curr = ss_years[i].social_security_income
            self.assertGreaterEqual(curr, prev * 0.99,
                                    f"SS income dropped at age {ss_years[i].age}")

    def test_depletion_flag_set_on_high_spend(self):
        """A tiny portfolio with very high spending should eventually be depleted."""
        cfg = make_cfg(
            retirement_age=38,
            end_age=50,
            starting_wealth=100_000.0,
            annual_spending=200_000.0,
            monthly_contribution=0.0,
            social_security_start_age=999,
            cash_wedge_years=0.0,
        )
        depleted = any(sim.simulate_path(cfg, s).depleted_any_time for s in range(50))
        self.assertTrue(depleted, "Expected at least one depleted path with spending >> wealth")

    def test_no_depletion_on_very_low_spend(self):
        """A huge portfolio with trivial spending should never be depleted."""
        cfg = make_cfg(
            retirement_age=38,
            end_age=80,
            starting_wealth=10_000_000.0,
            annual_spending=1_000.0,
            monthly_contribution=0.0,
            target_equity_cagr=0.07,
        )
        depleted = any(sim.simulate_path(cfg, s).depleted_any_time for s in range(100))
        self.assertFalse(depleted, "Should never be depleted with $10M and $1k/yr spending")

    def test_one_time_event_inflow_increases_wealth(self):
        """A positive one-time event at a given age should increase final wealth."""
        age_of_event = 50
        cfg_base  = make_cfg(retirement_age=40, end_age=60, one_time_events={})
        cfg_event = make_cfg(retirement_age=40, end_age=60,
                             one_time_events={age_of_event: 1_000_000.0})
        finals_base  = [sim.simulate_path(cfg_base,  s).years[-1].total_wealth for s in range(20)]
        finals_event = [sim.simulate_path(cfg_event, s).years[-1].total_wealth for s in range(20)]
        self.assertGreater(
            sum(finals_event) / len(finals_event),
            sum(finals_base)  / len(finals_base),
            "One-time inflow should increase average terminal wealth",
        )

    def test_charity_reduces_wealth(self):
        """Non-zero charity_pct should reduce terminal wealth vs. no charity."""
        cfg_no_charity   = make_cfg(retirement_age=40, end_age=70, charity_pct=0.0)
        cfg_with_charity = make_cfg(retirement_age=40, end_age=70, charity_pct=0.05)
        finals_no  = [sim.simulate_path(cfg_no_charity,   s).years[-1].total_wealth for s in range(20)]
        finals_yes = [sim.simulate_path(cfg_with_charity, s).years[-1].total_wealth for s in range(20)]
        self.assertGreater(
            sum(finals_no) / len(finals_no),
            sum(finals_yes) / len(finals_yes),
            "Charity should reduce terminal wealth",
        )


# ---------------------------------------------------------------------------
# RunSimulationTests
# ---------------------------------------------------------------------------

class RunSimulationTests(unittest.TestCase):

    def test_percentiles_monotonically_ordered(self):
        """p01 <= p25 <= median <= p75 <= p99 at every age."""
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
        """A lower target_equity_cagr should produce a lower median terminal wealth."""
        cfg_hi = make_cfg(simulations=500, target_equity_cagr=0.09, end_age=80)
        cfg_lo = make_cfg(simulations=500, target_equity_cagr=0.04, end_age=80)
        _, _, pct_hi, *_ = sim.run_simulation(cfg_hi)
        _, _, pct_lo, *_ = sim.run_simulation(cfg_lo)
        self.assertGreater(
            pct_hi[80]["median"], pct_lo[80]["median"],
            "Higher CAGR should produce higher median terminal wealth",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)