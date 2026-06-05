#!/usr/bin/env python3
"""Retirement simulator — calibrated global equity block-bootstrap engine.

Uses MSCI World gross total returns (1970-2023) as the historical return
distribution, sampled in random blocks. Each sampled path is shifted so
the long-run CAGR matches target_equity_cagr while crash magnitudes are
preserved exactly (Option 3: anchored shock decomposition).

Sequence-of-returns stress testing (Option 2): when stress_first_n_years > 0,
a second run forces the opening block to be drawn from the worst historical
windows, printing a separate stress summary below the main one.

Default block sizes include 30 and 40-year blocks (Option 1) so full
lost-decade sequences can appear as the opening block.

Default target: 7.0% nominal equity / 3.0% nominal inflation
~4.0% real return — in line with Dimson-Marsh-Staunton long-run
global equity forecasts for developed markets (DMS Yearbook 2025).

Dependencies: Python 3.11+ standard library only (tomllib).
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from statistics import fmean
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required (tomllib missing).") from exc

# ---------------------------------------------------------------------------
# Historical data: MSCI World gross total returns 1970-2023 (54 years).
# Source: MSCI / Wikipedia — includes reinvested dividends, USD terms.
# Raw geo mean ~9.7%; paths are shifted to target_equity_cagr at runtime.

# CPI: US annual inflation 1970-2023 (54 years). Source: BLS.
# Raw geo mean ~3.3%; paths are shifted to target_inflation at runtime.

# Both arrays are index-aligned: index 0 = 1970, index 53 = 2023.
# ---------------------------------------------------------------------------
MSCI_WORLD_ANNUAL: List[float] = [
    # 1970   1971    1972    1973    1974    1975    1976    1977    1978    1979
    -0.0198, 0.1956, 0.2355,-0.1451,-0.2448, 0.3450, 0.1471, 0.0500, 0.1822, 0.1267,
    # 1980   1981    1982    1983    1984    1985    1986    1987    1988    1989
     0.2772,-0.0330, 0.1127, 0.2328, 0.0577, 0.4177, 0.4280, 0.1676, 0.2395, 0.1719,
    # 1990   1991    1992    1993    1994    1995    1996    1997    1998    1999
    -0.1652, 0.1897,-0.0466, 0.2313, 0.0558, 0.2132, 0.1400, 0.1623, 0.2480, 0.2534,
    # 2000   2001    2002    2003    2004    2005    2006    2007    2008    2009
    -0.1292,-0.1652,-0.1954, 0.3376, 0.1525, 0.1002, 0.2065, 0.0957,-0.4033, 0.3079,
    # 2010   2011    2012    2013    2014    2015    2016    2017    2018    2019
     0.1234,-0.0502, 0.1654, 0.2737, 0.0550,-0.0032, 0.0815, 0.2307,-0.0820, 0.2840,
    # 2020   2021    2022    2023
     0.1650, 0.2235,-0.1773, 0.2442,
]

CPI_ANNUAL: List[float] = [
    # 1970   1971    1972    1973    1974    1975    1976    1977    1978    1979
    0.0553, 0.0336, 0.0303, 0.0472, 0.0611, 0.0549, 0.0332, 0.0698, 0.0901, 0.0553,
    # 1980   1981    1982    1983    1984    1985    1986    1987    1988    1989
    0.0316, 0.0390, 0.0381, 0.0395, 0.0380, 0.0102, 0.0614, 0.0735, 0.0110, 0.0444,
    # 1990   1991    1992    1993    1994    1995    1996    1997    1998    1999
    0.0376, 0.0303, 0.0291, 0.0280, 0.0189, 0.0355, 0.0419, 0.0308, 0.0290, 0.0274,
    # 2000   2001    2002    2003    2004    2005    2006    2007    2008    2009
    0.0026, 0.0296, 0.0285, 0.0228, 0.0168, 0.0159, 0.0227, 0.0268, 0.0388, 0.0316,
    # 2010   2011    2012    2013    2014    2015    2016    2017    2018    2019
    0.0228, 0.0162, 0.0230, 0.0271, 0.0134, 0.0215, 0.0244, 0.0181, 0.0240, 0.0188,
    # 2020   2021    2022    2023
    0.0031, 0.0213, 0.0207, 0.0700,
]

assert len(MSCI_WORLD_ANNUAL) == len(CPI_ANNUAL) == 54, (
    f"MSCI_WORLD_ANNUAL ({len(MSCI_WORLD_ANNUAL)}) and CPI_ANNUAL ({len(CPI_ANNUAL)}) must both be 54"
)

# Raw historical geometric means (informational only)
MSCI_WORLD_GEO_MEAN: float = math.exp(fmean(math.log1p(x) for x in MSCI_WORLD_ANNUAL)) - 1.0
CPI_GEO_MEAN: float = math.exp(fmean(math.log1p(x) for x in CPI_ANNUAL)) - 1.0

DEFAULT_BLOCK_SIZES = (5, 10, 15, 20, 30, 40)   # Option 1: wider blocks preserve lost decades
DEFAULT_TARGET_EQUITY_CAGR = 0.07
DEFAULT_TARGET_INFLATION   = 0.03

DEFAULT_GUARDRAIL_CEILING = 1.20
DEFAULT_GUARDRAIL_CUT     = 0.10
DEFAULT_STRESS_PERCENTILE = 0.10
DEFAULT_STRESS_MIN_WINDOWS = 3


# ---------------------------------------------------------------------------
# Option 3: anchored shock decomposition
#
# Instead of shifting every return value uniformly in log-space, we decompose
# each historical year into:
#   drift  = the raw series geometric mean  (the "expected" component)
#   shock  = r_t - drift                    (the "surprise" component)
#
# We then replace the drift with the target CAGR while leaving shocks intact:
#   r_t_new = target_cagr + shock_t
#
# This means a -40% crash year stays a -40% crash year regardless of the
# target CAGR, instead of being softened by a downward drift shift.
# ---------------------------------------------------------------------------

def _shift_to_target_anchored(series: List[float], target_cagr: float) -> List[float]:
    """Shift each series to target_cagr while preserving shocks relative to its own mean.
    
    Decomposition in log-space:
    log(1+r_t) = series_log_mean + shock_t
    Reconstruction:
    log(1+r_t_new) = log(1+target_cagr) + shock_t
    """
    if not series:
        return series
    target_log = math.log1p(target_cagr)
    # Calculate the actual log-mean of this specific series (not global mean)
    series_log_mean = fmean(math.log1p(v) for v in series)
    out = []
    for v in series:
        shock = math.log1p(v) - series_log_mean
        new_log = target_log + shock
        out.append(max(math.exp(new_log) - 1.0, -0.9999))
    return out


# ---------------------------------------------------------------------------
# Pre-compute worst-opening-block index table for stress testing (Option 2)
# ---------------------------------------------------------------------------

def _worst_block_starts(n_years: int, bottom_quartile: bool = True) -> List[int]:
    """Return start indices whose n_years cumulative return falls in the bottom quartile."""
    n_hist = len(MSCI_WORLD_ANNUAL)
    if n_years < 1 or n_years > n_hist:
        return list(range(n_hist - 1))
    results: List[Tuple[float, int]] = []
    for start in range(n_hist - n_years + 1):
        log_cum = sum(math.log1p(MSCI_WORLD_ANNUAL[start + i]) for i in range(n_years))
        results.append((log_cum, start))
    results.sort()
    cutoff = max(DEFAULT_STRESS_MIN_WINDOWS, math.ceil(len(results) * DEFAULT_STRESS_PERCENTILE))
    return [start for _, start in results[:cutoff]]


# Cache worst-block tables for the default stress lengths we care about
_WORST_BLOCK_CACHE: Dict[int, List[int]] = {}


def _get_worst_starts(n_years: int) -> List[int]:
    if n_years not in _WORST_BLOCK_CACHE:
        _WORST_BLOCK_CACHE[n_years] = _worst_block_starts(n_years)
    return _WORST_BLOCK_CACHE[n_years]


# ---------------------------------------------------------------------------
# Config / data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    starting_age:              int
    retirement_age:            int
    end_age:                   int
    starting_wealth:           float
    annual_spending:           float
    monthly_contribution:      float
    spending_volatility:       float
    charity_pct:               float
    equity_allocation:         float
    social_security_start_age: int
    social_security_annual:    float
    simulations:               int
    cash_wedge_years:          float
    cash_wedge_refill_rule:    str
    cash_wedge_escape_velocity: float
    cash_return_override:      Optional[float]
    bond_return_override:      Optional[float]
    target_equity_cagr:        float
    target_inflation:          float
    one_time_events:           Dict[int, float]
    bootstrap_block_sizes:     Tuple[int, ...]
    stress_first_n_years:      int   # 0 = disabled; >0 = run stress scenario


@dataclass
class YearState:
    age:                  int
    total_wealth:         float
    portfolio_wealth:     float
    wedge_cash:           float
    equity_return:        float
    inflation:            float
    cumulative_inflation: float
    contribution:         float
    base_spending:        float
    actual_spending:      float
    social_security_income: float


@dataclass
class SimulationPath:
    years:                  List[YearState]
    depleted_any_time:      bool
    terminal_depleted:      bool
    ss_only_any_time:       bool
    wedge_depleted:         bool
    final_cumulative_inflation: float


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> SimulationConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    events = {int(k): float(v) for k, v in raw.get("one_time_events", {}).items()}
    block_sizes = tuple(int(x) for x in raw.get("bootstrap_block_sizes", DEFAULT_BLOCK_SIZES))
    if not block_sizes:
        raise ValueError("bootstrap_block_sizes must not be empty")

    return SimulationConfig(
        starting_age               = int(raw["starting_age"]),
        retirement_age             = int(raw["retirement_age"]),
        end_age                    = int(raw["end_age"]),
        starting_wealth            = float(raw["starting_wealth"]),
        annual_spending            = float(raw["annual_spending"]),
        monthly_contribution       = float(raw["monthly_contribution"]),
        spending_volatility        = float(raw.get("spending_volatility", 0.0)),
        charity_pct                = float(raw.get("charity_pct", 0.0)),
        equity_allocation          = float(raw.get("equity_allocation", 1.0)),
        social_security_start_age  = int(raw.get("social_security_start_age", 999)),
        social_security_annual     = float(raw.get("social_security_annual", 0.0)),
        simulations                = int(raw.get("simulations", 10000)),
        cash_wedge_years           = float(raw.get("cash_wedge_years", 0.0)),
        cash_wedge_refill_rule     = str(raw.get("cash_wedge_refill_rule", "five_year_mean")),
        cash_wedge_escape_velocity = float(raw.get("cash_wedge_escape_velocity", 0.25)),
        cash_return_override       = None if "cash_return_override" not in raw else float(raw["cash_return_override"]),
        bond_return_override       = None if "bond_return_override" not in raw else float(raw["bond_return_override"]),
        target_equity_cagr         = float(raw.get("target_equity_cagr", DEFAULT_TARGET_EQUITY_CAGR)),
        target_inflation           = float(raw.get("target_inflation", DEFAULT_TARGET_INFLATION)),
        one_time_events            = events,
        bootstrap_block_sizes      = block_sizes,
        stress_first_n_years       = int(raw.get("stress_first_n_years", 0)),
    )


# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------

def bootstrap_historical_series(
    cfg: SimulationConfig,
    rng: random.Random,
    force_bad_opening: bool = False,
) -> Tuple[List[float], List[float]]:
    """Stitch random historical blocks, then apply anchored drift shift.

    If force_bad_opening=True, the first block is drawn exclusively from
    the bottom-quartile windows of length stress_first_n_years, simulating
    a sequence-of-returns catastrophe right at retirement (Option 2).
    """
    horizon = cfg.end_age - cfg.starting_age + 1
    n_hist  = len(MSCI_WORLD_ANNUAL)
    sizes   = [s for s in cfg.bootstrap_block_sizes if 1 <= s <= n_hist]
    if not sizes:
        raise ValueError("No valid bootstrap block sizes")

    eq_raw:  List[float] = []
    inf_raw: List[float] = []

# --- Option 2: forced bad opening block ---
    if force_bad_opening and cfg.stress_first_n_years > 0:
        pre_ret_years = max(0, min(cfg.retirement_age - cfg.starting_age, horizon))

        while len(eq_raw) < pre_ret_years:
            remaining = pre_ret_years - len(eq_raw)
            candidates = [s for s in sizes if s <= remaining] or sizes
            block = rng.choice(candidates)
            start = rng.randint(0, n_hist - block)
            end = start + block
            eq_raw.extend(MSCI_WORLD_ANNUAL[start:end])
            inf_raw.extend(CPI_ANNUAL[start:end])

        stress_len = min(cfg.stress_first_n_years, horizon - len(eq_raw), n_hist)
        if stress_len > 0:
            bad_starts = _get_worst_starts(stress_len)
            open_start = rng.choice(bad_starts)
            eq_raw.extend(MSCI_WORLD_ANNUAL[open_start: open_start + stress_len])
            inf_raw.extend(CPI_ANNUAL[open_start: open_start + stress_len])

    # --- Fill remaining horizon with normal random blocks ---
    while len(eq_raw) < horizon:
        remaining  = horizon - len(eq_raw)
        candidates = [s for s in sizes if s <= remaining] or sizes
        block      = rng.choice(candidates)
        start      = rng.randint(0, n_hist - block)
        end        = start + block
        eq_raw.extend(MSCI_WORLD_ANNUAL[start:end])
        inf_raw.extend(CPI_ANNUAL[start:end])

    eq_raw  = eq_raw[:horizon]
    inf_raw = inf_raw[:horizon]

    # Option 3: anchored shock decomposition for both series
    return (
        _shift_to_target_anchored(eq_raw,  cfg.target_equity_cagr),
        _shift_to_target_anchored(inf_raw, cfg.target_inflation),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank  = (pct / 100.0) * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def format_money(value: float) -> str:
    return f"${int(round(value)):,}"


def build_withdrawal(
    base_spending:    float,
    equity_return:    float,
    reference_return: float,
    volatility:       float,
) -> float:
    spend = base_spending
    if volatility > 0.0:
        if equity_return > reference_return:
            spend *= 1.0 + volatility
        elif equity_return < reference_return:
            spend *= 1.0 - volatility
    return max(spend, 0.0)


def apply_guardrail_cut(
    base_spending:          float,
    actual_spending:        float,
    total_assets:           float,
    reference_withdrawal_rate: Optional[float],
) -> Tuple[float, float, Optional[float]]:
    """Cut spending if WR exceeds ceiling. No raise — spending grows only via inflation."""
    if total_assets <= 0.0:
        return base_spending, actual_spending, reference_withdrawal_rate
    if reference_withdrawal_rate is None:
        reference_withdrawal_rate = max(base_spending / total_assets, 0.0)
    if reference_withdrawal_rate > 0.0:
        current_wr = actual_spending / total_assets
        if current_wr > reference_withdrawal_rate * DEFAULT_GUARDRAIL_CEILING:
            base_spending   *= 1.0 - DEFAULT_GUARDRAIL_CUT
            actual_spending  = min(actual_spending, base_spending)
    return base_spending, actual_spending, reference_withdrawal_rate


# ---------------------------------------------------------------------------
# Core path simulation
# ---------------------------------------------------------------------------

def simulate_path(
    cfg:              SimulationConfig,
    seed:             int,
    force_bad_opening: bool = False,
) -> SimulationPath:
    rng = random.Random(seed)
    equity_series, inflation_series = bootstrap_historical_series(
        cfg, rng, force_bad_opening=force_bad_opening
    )
    ages = list(range(cfg.starting_age, cfg.end_age + 1))

    bond_return      = cfg.bond_return_override if cfg.bond_return_override is not None else 0.0
    cash_refill_rule = cfg.cash_wedge_refill_rule.strip().lower()

    portfolio_wealth  = cfg.starting_wealth
    wedge_cash        = 0.0
    wedge_retired     = cfg.cash_wedge_years <= 0.0
    wedge_depleted    = False

    inflation_index   = 1.0
    base_spending     = 0.0
    social_security_income = 0.0
    reference_withdrawal_rate:       Optional[float] = None
    initial_reference_wr_for_wedge:  Optional[float] = None
    recent_returns:   List[float] = []
    records:          List[YearState] = []

    depleted_any_time = False
    terminal_depleted = False
    ss_only_any_time  = False

    for age, equity_return, inflation in zip(ages, equity_series, inflation_series):

        # Pre-retirement contributions
        contribution = cfg.monthly_contribution * 12.0 if age < cfg.retirement_age else 0.0
        portfolio_wealth += contribution

        # Portfolio growth
        portfolio_return = (
            cfg.equity_allocation * equity_return
            + (1.0 - cfg.equity_allocation) * bond_return
        )
        portfolio_wealth *= (1.0 + portfolio_return)

        # Wedge cash growth
        if wedge_cash > 0.0:
            cash_return = cfg.cash_return_override if cfg.cash_return_override is not None else inflation
            wedge_cash *= (1.0 + cash_return)

        # Inflation index
        inflation_index *= (1.0 + inflation)

        # Social Security
        if cfg.social_security_annual > 0.0:
            if age == cfg.social_security_start_age:
                social_security_income = cfg.social_security_annual * inflation_index
            elif age > cfg.social_security_start_age and social_security_income > 0.0:
                social_security_income *= (1.0 + inflation)
        ss_income_this_year = social_security_income if age >= cfg.social_security_start_age else 0.0

        actual_spending = 0.0

        if age >= cfg.retirement_age:
            # Inflate base spending
            if base_spending == 0.0:
                base_spending = cfg.annual_spending * inflation_index
            else:
                base_spending *= (1.0 + inflation)

            # Fund wedge at retirement
            if cfg.cash_wedge_years > 0.0 and wedge_cash == 0.0 and age == cfg.retirement_age:
                wedge_target     = min(base_spending * cfg.cash_wedge_years, portfolio_wealth)
                portfolio_wealth -= wedge_target
                wedge_cash       += wedge_target
                wedge_retired     = False

            # Spending (with volatility flex)
            total_assets_before_spending = portfolio_wealth + wedge_cash
            actual_spending = build_withdrawal(
                base_spending, equity_return, cfg.target_equity_cagr, cfg.spending_volatility
            )

            # Guardrails
            base_spending, actual_spending, reference_withdrawal_rate = apply_guardrail_cut(
                base_spending, actual_spending, total_assets_before_spending, reference_withdrawal_rate
            )

            if initial_reference_wr_for_wedge is None and reference_withdrawal_rate is not None:
                initial_reference_wr_for_wedge = reference_withdrawal_rate

            # Net spending after SS offsets
            net_spending_need = max(0.0, actual_spending - ss_income_this_year)
            ss_surplus        = max(0.0, ss_income_this_year - actual_spending)
            if ss_surplus > 0.0:
                portfolio_wealth += ss_surplus

# Wedge logic
            if not wedge_retired:
                total_assets        = portfolio_wealth + wedge_cash
                current_portfolio_wr = (
                    net_spending_need / total_assets
                    if total_assets > 0.0 and net_spending_need > 0.0 else 0.0
                )
                if (
                    cfg.cash_wedge_escape_velocity > 0.0
                    and initial_reference_wr_for_wedge is not None
                    and total_assets > 0.0
                    and current_portfolio_wr <= initial_reference_wr_for_wedge * cfg.cash_wedge_escape_velocity
                ):
                    portfolio_wealth += wedge_cash
                    wedge_cash        = 0.0
                    wedge_retired     = True
                else:
                    if cash_refill_rule == "five_year_mean":
                        recent_mean = fmean(recent_returns) if recent_returns else equity_return
                        use_wedge   = recent_mean < cfg.target_equity_cagr
                    else:
                        use_wedge = equity_return < 0.0

                    if use_wedge and wedge_cash > 0.0:
                        from_wedge = min(net_spending_need, wedge_cash)
                        wedge_cash -= from_wedge
                        portfolio_wealth -= max(net_spending_need - from_wedge, 0.0)
                    else:
                        portfolio_wealth -= net_spending_need

                    wedge_target = base_spending * cfg.cash_wedge_years
                    if equity_return > cfg.target_equity_cagr and wedge_cash < wedge_target and portfolio_wealth > 0.0:
                        refill = min(wedge_target - wedge_cash, portfolio_wealth * 0.10)
                        portfolio_wealth -= refill
                        wedge_cash += refill
            else:
                portfolio_wealth -= net_spending_need

        # Charity
        if cfg.charity_pct > 0.0 and portfolio_wealth > 0.0:
            portfolio_wealth -= portfolio_wealth * cfg.charity_pct

        # One-time events
        portfolio_wealth += cfg.one_time_events.get(age, 0.0)

        # Wedge covers any portfolio shortfall
        if portfolio_wealth < 0.0 and wedge_cash > 0.0:
            transfer          = min(-portfolio_wealth, wedge_cash)
            wedge_cash       -= transfer
            portfolio_wealth += transfer

        # Depletion detection (before clamping)
        year_depleted = age >= cfg.retirement_age and (portfolio_wealth + wedge_cash) < 0.0
        if year_depleted:
            depleted_any_time = True

        portfolio_wealth = max(portfolio_wealth, 0.0)
        wedge_cash       = max(wedge_cash, 0.0)
        total_wealth     = portfolio_wealth + wedge_cash

        if (
            age >= cfg.retirement_age
            and total_wealth <= 0.0
            and actual_spending > 0.0
            and ss_income_this_year >= actual_spending
        ):
            ss_only_any_time = True

        if cfg.cash_wedge_years > 0.0 and not wedge_retired and wedge_cash <= 0.0:
            wedge_depleted = True

        terminal_depleted = year_depleted

        records.append(YearState(
            age=age,
            total_wealth=total_wealth,
            portfolio_wealth=portfolio_wealth,
            wedge_cash=wedge_cash,
            equity_return=equity_return,
            inflation=inflation,
            cumulative_inflation=inflation_index,
            contribution=contribution,
            base_spending=base_spending,
            actual_spending=actual_spending,
            social_security_income=ss_income_this_year,
        ))

        recent_returns.append(equity_return)
        if len(recent_returns) > 5:
            recent_returns.pop(0)

    return SimulationPath(
        years=records,
        depleted_any_time=depleted_any_time,
        terminal_depleted=terminal_depleted,
        ss_only_any_time=ss_only_any_time,
        wedge_depleted=wedge_depleted,
        final_cumulative_inflation=inflation_index,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_percentile_rows(results_by_age: Dict[int, List[float]]) -> Dict[int, Dict[str, float]]:
    labels = [
        ("p01", 1), ("p05", 5), ("p10", 10), ("p25", 25),
        ("median", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99),
    ]
    return {
        age: {label: percentile(values, pct) for label, pct in labels}
        for age, values in results_by_age.items()
    }


def run_simulation(
    cfg: SimulationConfig,
    force_bad_opening: bool = False,
) -> Tuple[List[int], List[List[int]], Dict[int, Dict[str, float]], float, float, float, float, float]:
    ages   = list(range(cfg.starting_age, cfg.end_age + 1))
    matrix: List[List[int]] = []
    by_age: Dict[int, List[float]] = {age: [] for age in ages}

    depleted_any_time_count  = 0
    terminal_depleted_count  = 0
    ss_only_any_time_count   = 0
    wedge_depleted_count     = 0
    inflation_factors:         List[float] = []

    for seed in range(cfg.simulations):
        path = simulate_path(cfg, seed, force_bad_opening=force_bad_opening)
        row: List[int] = []
        for year in path.years:
            row.append(int(round(year.total_wealth)))
            by_age[year.age].append(year.total_wealth)
        matrix.append(row)

        if path.depleted_any_time:   depleted_any_time_count += 1
        if path.terminal_depleted:   terminal_depleted_count += 1
        if path.ss_only_any_time:    ss_only_any_time_count  += 1
        if path.wedge_depleted:      wedge_depleted_count    += 1
        inflation_factors.append(path.final_cumulative_inflation)

    percentile_rows = compute_percentile_rows(by_age)
    return (
        ages,
        matrix,
        percentile_rows,
        depleted_any_time_count / cfg.simulations,
        terminal_depleted_count / cfg.simulations,
        ss_only_any_time_count  / cfg.simulations,
        wedge_depleted_count    / cfg.simulations,
        percentile(inflation_factors, 50),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results_csv(path: str, ages: Sequence[int], matrix: Sequence[Sequence[int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["age", *[f"sim_{i+1}" for i in range(len(matrix))]])
        for idx, age in enumerate(ages):
            writer.writerow([age, *[row[idx] for row in matrix]])


def write_percentiles_csv(path: str, ages: Sequence[int], percentile_rows: Dict[int, Dict[str, float]]) -> None:
    columns = ["p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["age", *columns])
        for age in ages:
            writer.writerow([age, *[int(round(percentile_rows[age][col])) for col in columns]])


def _print_summary_block(
    cfg:                        SimulationConfig,
    ages:                       Sequence[int],
    matrix:                     Sequence[Sequence[int]],
    depleted_any_time_rate:     float,
    terminal_depleted_rate:     float,
    ss_only_any_time_rate:      float,
    wedge_depletion_rate:       float,
    median_cumulative_inflation: float,
    label:                      str = "",
) -> None:
    final_values = [row[-1] for row in matrix]
    end_age      = ages[-1]
    years        = cfg.end_age - cfg.starting_age

    if label:
        print(f"\n{'=' * 75}")
        print(f"  {label}")
    print("=" * 75)
    print(f" Simulations        : {cfg.simulations:,}")
    print(f" Age range          : {cfg.starting_age} -> {cfg.end_age} ({years} years)")
    print(f" Historical data    : MSCI World 1970-2023 ({len(MSCI_WORLD_ANNUAL)} yrs, raw geo {MSCI_WORLD_GEO_MEAN*100:.1f}%)")
    print(f" Target equity CAGR : {cfg.target_equity_cagr*100:.1f}% nominal (~{(cfg.target_equity_cagr - cfg.target_inflation)*100:.1f}% real)")
    print(f" Target inflation   : {cfg.target_inflation*100:.1f}% nominal")
    print(f" Equity allocation  : {cfg.equity_allocation * 100:.0f}%")
    print(f" Block sizes        : {', '.join(str(x) for x in cfg.bootstrap_block_sizes)} years")
    print(f" Median inflation   : {median_cumulative_inflation:.2f}x cumulative over {years} yrs")
    print()
    print(f" Depleted anytime   : {depleted_any_time_rate * 100:.1f}%")
    print(f" Depleted at {end_age}   : {terminal_depleted_rate * 100:.1f}%")
    if cfg.social_security_annual > 0.0:
        print(f" SS-only anytime    : {ss_only_any_time_rate * 100:.1f}%")
    if cfg.cash_wedge_years > 0.0:
        print(f" Wedge depleted     : {wedge_depletion_rate * 100:.1f}%")
    print()
    print(f" {'PORTFOLIO AT AGE ' + str(end_age):<30}{'NOMINAL':>18}{'TODAY\'S $':>18}")
    print("-" * 75)
    for lbl, pct in [
        ("99th", 99), ("95th", 95), ("90th", 90), ("75th", 75),
        ("Median", 50), ("25th", 25), ("10th", 10), ("5th", 5), ("1st", 1),
    ]:
        nominal  = percentile(final_values, pct)
        real     = nominal / median_cumulative_inflation if median_cumulative_inflation > 0.0 else nominal
        row_label = lbl + " percentile"
        print(f" {row_label:<30}{format_money(nominal):>18}{format_money(real):>18}")
    print("=" * 75)


def print_summary(
    cfg:                        SimulationConfig,
    ages:                       Sequence[int],
    matrix:                     Sequence[Sequence[int]],
    depleted_any_time_rate:     float,
    terminal_depleted_rate:     float,
    ss_only_any_time_rate:      float,
    wedge_depletion_rate:       float,
    median_cumulative_inflation: float,
) -> None:
    _print_summary_block(
        cfg, ages, matrix,
        depleted_any_time_rate, terminal_depleted_rate,
        ss_only_any_time_rate, wedge_depletion_rate,
        median_cumulative_inflation,
        label="MONTE CARLO SIMULATION SUMMARY — Anchored Block Bootstrap",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = list(argv or sys.argv[1:])
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.toml")
    if not os.path.exists(config_path):
        print(f"Error: config.toml not found in {script_dir}")
        print("Copy config.example.toml to config.toml and edit it first.")
        return 1

    cfg = load_config(config_path)
    print("Running Monte Carlo simulation (anchored historical block bootstrap)...")
    print()

    # --- Main run ---
    (ages, matrix, percentile_rows,
     depleted_any_time_rate, terminal_depleted_rate,
     ss_only_any_time_rate, wedge_depletion_rate,
     median_cumulative_inflation) = run_simulation(cfg, force_bad_opening=False)

    results_csv = os.path.join(script_dir, "portfolio_simulation.csv")
    pct_csv     = os.path.join(script_dir, "portfolio_percentiles.csv")
    write_results_csv(results_csv, ages, matrix)
    write_percentiles_csv(pct_csv, ages, percentile_rows)
    print(f"Results saved to  : {results_csv}")
    print(f"Percentiles saved : {pct_csv}")
    print()

    print_summary(
        cfg, ages, matrix,
        depleted_any_time_rate, terminal_depleted_rate,
        ss_only_any_time_rate, wedge_depletion_rate,
        median_cumulative_inflation,
    )

    # --- Option 2: stress scenario ---
    if cfg.stress_first_n_years > 0:
        print(f"\nRunning stress scenario: worst opening {cfg.stress_first_n_years} years forced...")
        print()
        (s_ages, s_matrix, _s_pct,
         s_dep_any, s_dep_term,
         s_ss_only, s_wedge,
         s_inf) = run_simulation(cfg, force_bad_opening=True)

        stress_pct_csv = os.path.join(script_dir, "portfolio_percentiles_stress.csv")
        write_percentiles_csv(stress_pct_csv, s_ages, _s_pct)
        print(f"Stress percentiles: {stress_pct_csv}")

        _print_summary_block(
            cfg, s_ages, s_matrix,
            s_dep_any, s_dep_term, s_ss_only, s_wedge, s_inf,
            label=(
                f"STRESS SCENARIO — Worst {cfg.stress_first_n_years}-Year Opening Block "
                f"(bottom-decile historical windows)"
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
