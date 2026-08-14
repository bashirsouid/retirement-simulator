#!/usr/bin/env python3
"""Retirement simulator — calibrated global equity block-bootstrap engine.

Uses MSCI World gross total returns (1970-2023) as the historical return
distribution, sampled in random blocks. Each sampled path is shifted so
the long-run CAGR matches target_equity_cagr while crash magnitudes are
preserved relative to the global historical mean.

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

MSCI_WORLD_ANNUAL: List[float] = [
    -0.0198, 0.1956, 0.2355,-0.1451,-0.2448, 0.3450, 0.1471, 0.0500, 0.1822, 0.1267,
     0.2772,-0.0330, 0.1127, 0.2328, 0.0577, 0.4177, 0.4280, 0.1676, 0.2395, 0.1719,
    -0.1652, 0.1897,-0.0466, 0.2313, 0.0558, 0.2132, 0.1400, 0.1623, 0.2480, 0.2534,
    -0.1292,-0.1652,-0.1954, 0.3376, 0.1525, 0.1002, 0.2065, 0.0957,-0.4033, 0.3079,
     0.1234,-0.0502, 0.1654, 0.2737, 0.0550,-0.0032, 0.0815, 0.2307,-0.0820, 0.2840,
     0.1650, 0.2235,-0.1773, 0.2442,
]

CPI_ANNUAL: List[float] = [
     0.055,  0.033,  0.034,  0.087,  0.124,  0.070,  0.049,  0.067,  0.076,  0.113,
     0.135,  0.103,  0.061,  0.032,  0.043,  0.036,  0.019,  0.037,  0.041,  0.048,
     0.054,  0.042,  0.030,  0.030,  0.026,  0.028,  0.029,  0.023,  0.016,  0.022,
     0.034,  0.028,  0.016,  0.023,  0.027,  0.034,  0.032,  0.029,  0.038, -0.004,
     0.016,  0.032,  0.021,  0.015,  0.016,  0.001,  0.013,  0.021,  0.024,  0.018,
     0.012,  0.047,  0.080,  0.041,
]

assert len(MSCI_WORLD_ANNUAL) == len(CPI_ANNUAL) == 54, (
    f"MSCI_WORLD_ANNUAL ({len(MSCI_WORLD_ANNUAL)}) and CPI_ANNUAL ({len(CPI_ANNUAL)}) must both be 54"
)

MSCI_WORLD_GEO_MEAN: float = math.exp(fmean(math.log1p(x) for x in MSCI_WORLD_ANNUAL)) - 1.0
CPI_GEO_MEAN: float = math.exp(fmean(math.log1p(x) for x in CPI_ANNUAL)) - 1.0
MSCI_WORLD_GEO_LOG_MEAN: float = fmean(math.log1p(x) for x in MSCI_WORLD_ANNUAL)
CPI_GEO_LOG_MEAN: float = fmean(math.log1p(x) for x in CPI_ANNUAL)

DEFAULT_BLOCK_SIZES = (4, 5, 6, 8)
DEFAULT_TARGET_EQUITY_CAGR = 0.07
DEFAULT_TARGET_INFLATION   = 0.03

DEFAULT_GUARDRAIL_CEILING = 1.20
DEFAULT_GUARDRAIL_RESTORE = 1.00
DEFAULT_GUARDRAIL_CUT     = 0.10
DEFAULT_STRESS_PERCENTILE = 0.10
DEFAULT_STRESS_MIN_WINDOWS = 3
DEFAULT_MEDICARE_AGE = 65


def _shift_to_target_anchored(series: List[float], target_cagr: float, global_log_mean: float) -> List[float]:
    if not series:
        return series
    target_log = math.log1p(target_cagr)
    out = []
    for v in series:
        shock = math.log1p(v) - global_log_mean
        new_log = target_log + shock
        out.append(max(math.exp(new_log) - 1.0, -0.9999))
    return out


def _worst_block_starts(n_years: int, bottom_quartile: bool = True) -> List[int]:
    n_hist = len(MSCI_WORLD_ANNUAL)
    if n_years < 1 or n_years > n_hist:
        return list(range(max(n_hist - 1, 1)))
    results: List[Tuple[float, int]] = []
    for start in range(n_hist - n_years + 1):
        log_cum = sum(math.log1p(MSCI_WORLD_ANNUAL[start + i]) for i in range(n_years))
        results.append((log_cum, start))
    results.sort()
    cutoff = max(DEFAULT_STRESS_MIN_WINDOWS, math.ceil(len(results) * DEFAULT_STRESS_PERCENTILE))
    return [start for _, start in results[:cutoff]]


_WORST_BLOCK_CACHE: Dict[int, List[int]] = {}


def _get_worst_starts(n_years: int) -> List[int]:
    if n_years not in _WORST_BLOCK_CACHE:
        _WORST_BLOCK_CACHE[n_years] = _worst_block_starts(n_years)
    return _WORST_BLOCK_CACHE[n_years]


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
    stress_first_n_years:      int
    withdrawal_tax_rate:       float
    inflate_contributions:     bool
    medicare_age:              int
    pre_medicare_extra_annual: float


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
    wedge_depleted:         bool
    final_cumulative_inflation: float


def load_config(path: str) -> SimulationConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    events = {int(k): float(v) for k, v in raw.get("one_time_events", {}).items()}
    block_sizes = tuple(int(x) for x in raw.get("bootstrap_block_sizes", DEFAULT_BLOCK_SIZES))
    if not block_sizes:
        raise ValueError("bootstrap_block_sizes must not be empty")

    cfg = SimulationConfig(
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
        withdrawal_tax_rate        = float(raw.get("withdrawal_tax_rate", 0.0)),
        inflate_contributions      = bool(raw.get("inflate_contributions", True)),
        medicare_age               = int(raw.get("medicare_age", DEFAULT_MEDICARE_AGE)),
        pre_medicare_extra_annual  = float(raw.get("pre_medicare_extra_annual", 0.0)),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: SimulationConfig) -> None:
    if cfg.end_age < cfg.starting_age:
        raise ValueError("end_age must be greater than or equal to starting_age")
    if not 0.0 <= cfg.equity_allocation <= 1.0:
        raise ValueError("equity_allocation must be between 0.0 and 1.0")
    if cfg.equity_allocation < 1.0 and cfg.bond_return_override is None:
        raise ValueError(
            "bond_return_override is required when equity_allocation is below 1.0; "
            "the simulator has no historical bond-return series"
        )
    if cfg.simulations < 1:
        raise ValueError("simulations must be at least 1")
    if cfg.cash_wedge_years < 0.0:
        raise ValueError("cash_wedge_years must not be negative")
    if cfg.target_equity_cagr <= -1.0 or cfg.target_inflation <= -1.0:
        raise ValueError("target return and inflation values must be greater than -100%")
    if not 0.0 <= cfg.withdrawal_tax_rate < 1.0:
        raise ValueError("withdrawal_tax_rate must be in [0.0, 1.0)")
    if cfg.pre_medicare_extra_annual < 0.0:
        raise ValueError("pre_medicare_extra_annual must not be negative")
    if cfg.medicare_age < 0:
        raise ValueError("medicare_age must not be negative")


def bootstrap_historical_series(
    cfg: SimulationConfig,
    rng: random.Random,
    force_bad_opening: bool = False,
) -> Tuple[List[float], List[float]]:
    horizon = cfg.end_age - cfg.starting_age + 1
    n_hist  = len(MSCI_WORLD_ANNUAL)
    sizes   = [s for s in cfg.bootstrap_block_sizes if 1 <= s <= n_hist]
    if not sizes:
        raise ValueError("No valid bootstrap block sizes")

    eq_raw:  List[float] = []
    inf_raw: List[float] = []

    while len(eq_raw) < horizon:
        remaining = horizon - len(eq_raw)
        candidates = [s for s in sizes if s <= remaining] or sizes
        block = rng.choice(candidates)
        start = rng.randint(0, n_hist - block)
        end = start + block
        eq_raw.extend(MSCI_WORLD_ANNUAL[start:end])
        inf_raw.extend(CPI_ANNUAL[start:end])

    eq_raw  = eq_raw[:horizon]
    inf_raw = inf_raw[:horizon]

    if force_bad_opening and cfg.stress_first_n_years > 0:
        pre_ret_years = max(0, min(cfg.retirement_age - cfg.starting_age, horizon))
        stress_len = min(cfg.stress_first_n_years, horizon - pre_ret_years, n_hist)

        if stress_len > 0:
            bad_starts = _get_worst_starts(stress_len)
            stress_rng = random.Random(rng.randint(0, 2**31 - 1))
            open_start = stress_rng.choice(bad_starts)
            ret_start = pre_ret_years
            eq_raw = eq_raw[:ret_start] + MSCI_WORLD_ANNUAL[open_start:open_start + stress_len] + eq_raw[ret_start + stress_len:]
            inf_raw = inf_raw[:ret_start] + CPI_ANNUAL[open_start:open_start + stress_len] + inf_raw[ret_start + stress_len:]

    return (
        _shift_to_target_anchored(eq_raw,  cfg.target_equity_cagr, MSCI_WORLD_GEO_LOG_MEAN),
        _shift_to_target_anchored(inf_raw, cfg.target_inflation, CPI_GEO_LOG_MEAN),
    )


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


def real_wealth(nominal: float, cumulative_inflation: float) -> float:
    if cumulative_inflation <= 0.0:
        return nominal
    return nominal / cumulative_inflation


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


def apply_guardrail(
    lifestyle_base: float,
    base_spending: float,
    actual_spending: float,
    total_assets: float,
    reference_withdrawal_rate: Optional[float],
) -> Tuple[float, float, Optional[float], bool]:
    """Cut lifestyle in a downturn; restore it when the WR is healthy again.

    Returns (base_spending, actual_spending, reference_wr, restored).
    restored=True means the caller should recompute actual_spending from the
    restored base (volatility flex on the full lifestyle).
    """
    if total_assets <= 0.0:
        return base_spending, actual_spending, reference_withdrawal_rate, False
    if reference_withdrawal_rate is None:
        reference_withdrawal_rate = max(lifestyle_base / total_assets, 0.0)
    if reference_withdrawal_rate <= 0.0:
        return base_spending, actual_spending, reference_withdrawal_rate, False

    current_wr = actual_spending / total_assets
    floor = lifestyle_base * (1.0 - DEFAULT_GUARDRAIL_CUT)  # Percent of full lifestyle
    if current_wr > reference_withdrawal_rate * DEFAULT_GUARDRAIL_CEILING:
        base_spending = floor
        actual_spending = min(actual_spending, base_spending)
        return base_spending, actual_spending, reference_withdrawal_rate, False
    if current_wr <= reference_withdrawal_rate * DEFAULT_GUARDRAIL_RESTORE and base_spending < lifestyle_base:
        return lifestyle_base, actual_spending, reference_withdrawal_rate, True
    return base_spending, actual_spending, reference_withdrawal_rate, False


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
    wedge_initialized = cfg.cash_wedge_years <= 0.0
    wedge_depleted    = False

    inflation_index   = 1.0
    lifestyle_base    = 0.0
    base_spending     = 0.0
    social_security_income = 0.0
    reference_withdrawal_rate:       Optional[float] = None
    initial_reference_wr_for_wedge:  Optional[float] = None
    recent_returns:   List[float] = []
    records:          List[YearState] = []

    depleted_any_time = False
    terminal_depleted = False

    for age, equity_return, inflation in zip(ages, equity_series, inflation_series):

        contribution = cfg.monthly_contribution * 12.0 if age < cfg.retirement_age else 0.0
        if contribution > 0.0 and cfg.inflate_contributions:
            contribution *= inflation_index

        if cfg.social_security_annual > 0.0 and age >= cfg.social_security_start_age:
            social_security_income = cfg.social_security_annual * inflation_index
        else:
            social_security_income = 0.0
        ss_income_this_year = social_security_income

        actual_spending = 0.0

        if age >= cfg.retirement_age:
            if lifestyle_base == 0.0:
                lifestyle_base = cfg.annual_spending * inflation_index
                base_spending = lifestyle_base

            if (
                cfg.cash_wedge_years > 0.0
                and not wedge_initialized
                and age >= cfg.retirement_age
            ):
                wedge_target     = min(base_spending * cfg.cash_wedge_years, portfolio_wealth)
                portfolio_wealth -= wedge_target
                wedge_cash       += wedge_target
                wedge_retired     = False
                wedge_initialized = True

            total_assets_before_spending = portfolio_wealth + wedge_cash
            actual_spending = build_withdrawal(
                base_spending, equity_return, cfg.target_equity_cagr, cfg.spending_volatility
            )

            base_spending, actual_spending, reference_withdrawal_rate, restored = apply_guardrail(
                lifestyle_base, base_spending, actual_spending,
                total_assets_before_spending, reference_withdrawal_rate,
            )
            if restored:
                actual_spending = build_withdrawal(
                    base_spending, equity_return, cfg.target_equity_cagr, cfg.spending_volatility
                )

            if age < cfg.medicare_age and cfg.pre_medicare_extra_annual > 0.0:
                actual_spending += cfg.pre_medicare_extra_annual * inflation_index

            if initial_reference_wr_for_wedge is None and reference_withdrawal_rate is not None:
                initial_reference_wr_for_wedge = reference_withdrawal_rate

            net_spending_need = max(0.0, actual_spending - ss_income_this_year)
            ss_surplus        = max(0.0, ss_income_this_year - actual_spending)
            if ss_surplus > 0.0:
                portfolio_wealth += ss_surplus

            if net_spending_need > 0.0 and cfg.withdrawal_tax_rate > 0.0:
                net_spending_need /= (1.0 - cfg.withdrawal_tax_rate)

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

        if cfg.charity_pct > 0.0 and portfolio_wealth > 0.0:
            portfolio_wealth -= portfolio_wealth * cfg.charity_pct

        portfolio_wealth += cfg.one_time_events.get(age, 0.0) * inflation_index
        portfolio_wealth += contribution

        if portfolio_wealth < 0.0 and wedge_cash > 0.0:
            transfer          = min(-portfolio_wealth, wedge_cash)
            wedge_cash       -= transfer
            portfolio_wealth += transfer

        year_depleted = age >= cfg.retirement_age and (portfolio_wealth + wedge_cash) <= 0.0
        if year_depleted:
            depleted_any_time = True

        portfolio_wealth = max(portfolio_wealth, 0.0)
        wedge_cash       = max(wedge_cash, 0.0)

        portfolio_return = (
            cfg.equity_allocation * equity_return
            + (1.0 - cfg.equity_allocation) * bond_return
        )
        portfolio_wealth *= 1.0 + portfolio_return
        if wedge_cash > 0.0:
            cash_return = cfg.cash_return_override if cfg.cash_return_override is not None else inflation
            wedge_cash *= 1.0 + cash_return

        total_wealth = portfolio_wealth + wedge_cash
        if year_depleted:
            terminal_depleted = True
        else:
            terminal_depleted = age >= cfg.retirement_age and total_wealth <= 0.0

        if (
            age >= cfg.retirement_age
            and cfg.cash_wedge_years > 0.0
            and not wedge_retired
            and wedge_cash <= 0.0
        ):
            wedge_depleted = True

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

        inflation_index *= 1.0 + inflation
        if lifestyle_base > 0.0:
            lifestyle_base *= 1.0 + inflation
        if base_spending > 0.0:
            base_spending *= 1.0 + inflation

        recent_returns.append(equity_return)
        if len(recent_returns) > 5:
            recent_returns.pop(0)

    final_wedge_depleted = (
        cfg.cash_wedge_years > 0.0
        and not wedge_retired
        and wedge_cash <= 0.0
    )

    return SimulationPath(
        years=records,
        depleted_any_time=depleted_any_time,
        terminal_depleted=terminal_depleted,
        wedge_depleted=final_wedge_depleted,
        final_cumulative_inflation=inflation_index,
    )


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
) -> Tuple[List[int], List[List[int]], Dict[int, Dict[str, float]], Dict[int, Dict[str, float]], float, float, float, float]:
    ages   = list(range(cfg.starting_age, cfg.end_age + 1))
    matrix: List[List[int]] = []
    by_age: Dict[int, List[float]] = {age: [] for age in ages}
    by_age_real: Dict[int, List[float]] = {age: [] for age in ages}

    depleted_any_time_count  = 0
    terminal_depleted_count  = 0
    wedge_depleted_count     = 0
    inflation_factors:         List[float] = []

    for seed in range(cfg.simulations):
        path = simulate_path(cfg, seed, force_bad_opening=force_bad_opening)
        row: List[int] = []
        for year in path.years:
            row.append(int(round(year.total_wealth)))
            by_age[year.age].append(year.total_wealth)
            by_age_real[year.age].append(real_wealth(year.total_wealth, year.cumulative_inflation))
        matrix.append(row)

        if path.depleted_any_time:   depleted_any_time_count += 1
        if path.terminal_depleted:   terminal_depleted_count += 1
        if path.wedge_depleted:      wedge_depleted_count    += 1
        inflation_factors.append(path.final_cumulative_inflation)

    return (
        ages,
        matrix,
        compute_percentile_rows(by_age),
        compute_percentile_rows(by_age_real),
        depleted_any_time_count / cfg.simulations,
        terminal_depleted_count / cfg.simulations,
        wedge_depleted_count    / cfg.simulations,
        percentile(inflation_factors, 50),
    )


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
    real_percentile_rows:       Dict[int, Dict[str, float]],
    depleted_any_time_rate:     float,
    terminal_depleted_rate:     float,
    wedge_depletion_rate:       float,
    median_cumulative_inflation: float,
    label:                      str = "",
) -> None:
    final_values = [row[-1] for row in matrix]
    end_age      = ages[-1]
    years        = cfg.end_age - cfg.starting_age + 1
    real_final   = real_percentile_rows[end_age]

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
    if cfg.cash_wedge_years > 0.0:
        print(f" Wedge depleted     : {wedge_depletion_rate * 100:.1f}%")
    print()
    today_hdr = "TODAY'S $"
    print(f" {'PORTFOLIO AT AGE ' + str(end_age):<30}{'NOMINAL':>18}{today_hdr:>18}")
    print("-" * 75)
    for lbl, key in [
        ("99th", "p99"), ("95th", "p95"), ("90th", "p90"), ("75th", "p75"),
        ("Median", "median"), ("25th", "p25"), ("10th", "p10"), ("5th", "p05"), ("1st", "p01"),
    ]:
        pct = {"p99": 99, "p95": 95, "p90": 90, "p75": 75, "median": 50, "p25": 25, "p10": 10, "p05": 5, "p01": 1}[key]
        nominal  = percentile(final_values, pct)
        real     = real_final[key]
        row_label = lbl + " percentile"
        print(f" {row_label:<30}{format_money(nominal):>18}{format_money(real):>18}")
    print("=" * 75)


def print_summary(
    cfg:                        SimulationConfig,
    ages:                       Sequence[int],
    matrix:                     Sequence[Sequence[int]],
    real_percentile_rows:       Dict[int, Dict[str, float]],
    depleted_any_time_rate:     float,
    terminal_depleted_rate:     float,
    wedge_depletion_rate:       float,
    median_cumulative_inflation: float,
) -> None:
    _print_summary_block(
        cfg, ages, matrix, real_percentile_rows,
        depleted_any_time_rate, terminal_depleted_rate,
        wedge_depletion_rate,
        median_cumulative_inflation,
        label="MONTE CARLO SIMULATION SUMMARY — Anchored Block Bootstrap",
    )


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

    (ages, matrix, percentile_rows, real_percentile_rows,
     depleted_any_time_rate, terminal_depleted_rate,
     wedge_depletion_rate,
     median_cumulative_inflation) = run_simulation(cfg, force_bad_opening=False)

    results_csv = os.path.join(script_dir, "portfolio_simulation.csv")
    pct_csv     = os.path.join(script_dir, "portfolio_percentiles.csv")
    real_csv    = os.path.join(script_dir, "portfolio_percentiles_real.csv")
    write_results_csv(results_csv, ages, matrix)
    write_percentiles_csv(pct_csv, ages, percentile_rows)
    write_percentiles_csv(real_csv, ages, real_percentile_rows)
    print(f"Results saved to  : {results_csv}")
    print(f"Percentiles saved : {pct_csv}")
    print(f"Real $ percentiles: {real_csv}")
    print()

    print_summary(
        cfg, ages, matrix, real_percentile_rows,
        depleted_any_time_rate, terminal_depleted_rate,
        wedge_depletion_rate,
        median_cumulative_inflation,
    )

    if cfg.stress_first_n_years > 0:
        print(f"\nRunning stress scenario: worst opening {cfg.stress_first_n_years} years forced...")
        print()
        (s_ages, s_matrix, _s_pct, s_real,
         s_dep_any, s_dep_term,
         s_wedge,
         s_inf) = run_simulation(cfg, force_bad_opening=True)

        stress_sim_csv = os.path.join(script_dir, "portfolio_simulation_stress.csv")
        stress_pct_csv = os.path.join(script_dir, "portfolio_percentiles_stress.csv")
        stress_real_csv = os.path.join(script_dir, "portfolio_percentiles_real_stress.csv")
        write_results_csv(stress_sim_csv, s_ages, s_matrix)
        write_percentiles_csv(stress_pct_csv, s_ages, _s_pct)
        write_percentiles_csv(stress_real_csv, s_ages, s_real)
        print(f"Stress results saved to  : {stress_sim_csv}")
        print(f"Stress percentiles saved : {stress_pct_csv}")
        print(f"Stress real $ percentiles: {stress_real_csv}")

        _print_summary_block(
            cfg, s_ages, s_matrix, s_real,
            s_dep_any, s_dep_term, s_wedge, s_inf,
            label=(
                f"STRESS SCENARIO — Worst {cfg.stress_first_n_years}-Year Opening Block "
                f"(bottom-decile historical windows)"
            ),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
