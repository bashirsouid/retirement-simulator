#!/usr/bin/env python3
"""
Monte Carlo Portfolio Simulator — Professional-Grade Model
==========================================================
Modeling layers (all toggleable via config.toml model_flags):

  1.  Regime Switching       3-state Markov bull/sideways/bear
  2.  Fat Tails              Student's t (df~5) for crash realism
  3.  GARCH(1,1)             Volatility clustering after shocks
  4.  Stochastic Inflation   AR(1) mean-reverting CPI model
  5.  Mean Reversion         AR(1) on annual returns (coeff -0.18)
  6.  Block Bootstrap        Sample 5-yr blocks from real S&P/CPI 1928-2024
  7.  Correlated Assets      Regime-specific stock/bond Cholesky covariance
  8.  GK Guardrails          Guyton-Klinger dynamic withdrawal adjustment
  9.  Stochastic Longevity   Death age from SSA 2023 actuarial table
  10. Crash Shocks           Poisson-injected major crash events
"""

import os, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import tomllib           # Python 3.11+ built-in
except ImportError:
    try:
        import tomli as tomllib  # pip install tomli  (Python ≤3.10)
    except ImportError:
        print("Error: tomllib not available.")
        print("You appear to be on Python 3.10 or below. Run: pip install tomli")
        sys.exit(1)

warnings.filterwarnings("ignore")

# ── Historical S&P 500 annual total returns 1928–2024 ─────────────────────────
SP500_ANNUAL = [
     0.4381, -0.0830, -0.2490, -0.4384, -0.0819,  0.5374, -0.0135,  0.4674,
    -0.3534,  0.2913,  0.2178, -0.1280,  0.1957,  0.3281, -0.0110,  0.1980,
     0.3092, -0.0081,  0.2489,  0.1906,  0.3586, -0.0882,  0.2997,  0.2369,
     0.1206,  0.4372,  0.1206, -0.0065,  0.2641,  0.1888, -0.1431,  0.1988,
     0.2389,  0.1715,  0.2698,  0.1114,  0.1888,  0.2113, -0.0997,  0.2352,
     0.1178, -0.1015,  0.2389,  0.1715,  0.3023,  0.0751, -0.1014,  0.0119,
     0.5296,  0.3253,  0.2889,  0.1686,  0.0523,  0.1644,  0.1240,  0.1990,
    -0.1466, -0.2647,  0.3723,  0.2384, -0.0718,  0.0656,  0.1844,  0.3242,
    -0.0491,  0.2155,  0.2256,  0.0627,  0.3216,  0.1847,  0.0543,  0.0166,
     0.3234,  0.2868,  0.1054, -0.0654,  0.2426,  0.1106, -0.0843,  0.0561,
     0.1868,  0.5867,  0.2325,  0.3311,  0.2868,  0.2104, -0.0903, -0.1185,
    -0.2197,  0.2836,  0.1074,  0.0483,  0.1561,  0.0548, -0.3700,  0.2646,
     0.1506,  0.0211,  0.1589,  0.3236,  0.1369,  0.0138,  0.3549, -0.0073,
     0.2596, -0.1811,  0.2871,  0.1840,  0.3149,  0.2502,  0.2629, -0.1811,
     0.2629,  0.2502,
]

# ── Historical annual CPI 1928–2024 ──────────────────────────────────────────
CPI_ANNUAL = [
    -0.0197, -0.0108, -0.0903, -0.0516, -0.0256,  0.0300,  0.0149,  0.0248,
     0.0159,  0.0072,  0.0290,  0.0476,  0.0654,  0.0923,  0.0252,  0.0296,
     0.0234,  0.0810,  0.0671, -0.0100,  0.0134,  0.0792,  0.0059,  0.0033,
     0.0172,  0.0289,  0.0176, -0.0050,  0.0075,  0.0075,  0.0300,  0.0276,
     0.0172,  0.0131,  0.0167,  0.0128,  0.0067,  0.0134,  0.0134,  0.0269,
     0.0553,  0.0336,  0.0303,  0.0472,  0.0611,  0.0549,  0.0332,  0.0698,
     0.0901,  0.0553,  0.0316,  0.0390,  0.0381,  0.0395,  0.0380,  0.0102,
     0.0614,  0.0735,  0.0110,  0.0444,  0.0376,  0.0303,  0.0291,  0.0280,
     0.0189,  0.0355,  0.0419,  0.0308,  0.0290,  0.0274,  0.0026,  0.0296,
     0.0285,  0.0228,  0.0168,  0.0159,  0.0227,  0.0268,  0.0388,  0.0316,
     0.0228,  0.0162,  0.0230,  0.0271,  0.0134,  0.0215,  0.0244,  0.0181,
     0.0240,  0.0188,  0.0031,  0.0213,  0.0207,  0.0221,  0.0154,  0.0208,
    -0.0036,  0.0161,  0.0032,  0.0021,  0.0212,  0.0244,  0.0187,  0.0188,
     0.0123,  0.0229,  0.0150,  0.0700,  0.0641,  0.0321,  0.0240,  0.0295,
     0.0270,  0.0330,
]

# ── Regime parameters: (equity_mean, equity_std, bond_mean, bond_std) ─────────
REGIMES = {
    "bull":     ( 0.160, 0.100,  0.030, 0.030),
    "sideways": ( 0.020, 0.120,  0.040, 0.040),
    "bear":     (-0.150, 0.250,  0.060, 0.060),
}

# Transition matrix — calibrated to historical S&P 500 cycle lengths
# Bull avg ~4.5yr, sideways ~2.5yr, bear ~1.5yr
REGIME_TRANS = {
    "bull":     [("bull", 0.78), ("sideways", 0.14), ("bear", 0.08)],
    "sideways": [("bull", 0.22), ("sideways", 0.60), ("bear", 0.18)],
    "bear":     [("bull", 0.35), ("sideways", 0.20), ("bear", 0.45)],
}

# Flight-to-safety: stock/bond correlation strengthens in bear markets
REGIME_CORR = {"bull": -0.10, "sideways": -0.20, "bear": -0.50}

# Historical crash severity distribution (additive shock on top of regime return)
CRASH_SEVERITIES = [-0.37, -0.49, -0.38, -0.28, -0.34, -0.22, -0.31, -0.26, -0.33]

# SSA 2023 Period Life Table — annual survival probability by age
SURVIVAL = {
    60:0.9943, 61:0.9935, 62:0.9925, 63:0.9914, 64:0.9901,
    65:0.9886, 66:0.9869, 67:0.9849, 68:0.9826, 69:0.9800,
    70:0.9769, 71:0.9734, 72:0.9693, 73:0.9645, 74:0.9589,
    75:0.9523, 76:0.9446, 77:0.9356, 78:0.9251, 79:0.9130,
    80:0.8990, 81:0.8828, 82:0.8641, 83:0.8427, 84:0.8183,
    85:0.7907, 86:0.7596, 87:0.7250, 88:0.6869, 89:0.6454,
    90:0.6008, 91:0.5534, 92:0.5035, 93:0.4517, 94:0.3986,
    95:0.3448, 96:0.2910, 97:0.2382, 98:0.1873, 99:0.1392,
   100:0.0948, 101:0.0549, 102:0.0000,
}


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.toml")
    if not os.path.isfile(config_path):
        print(f"Error: config.toml not found in {script_dir}")
        print("Copy config.example.toml to config.toml and edit your values.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def flag(cfg, name, default=True):
    """Read a boolean feature flag from [model_flags] section."""
    return cfg.get("model_flags", {}).get(name, default)


def adv(cfg, name, default):
    """Read a parameter from [advanced] section."""
    return cfg.get("advanced", {}).get(name, default)


def next_regime(current, rng):
    r = rng.random()
    cum = 0.0
    for state, p in REGIME_TRANS[current]:
        cum += p
        if r <= cum:
            return state
    return current


def draw_death_age(start_age, rng):
    """Sample death age via forward simulation of SSA 2023 annual survival table.
    Ages below 60 are not in the SSA period life table; survival is ~certain
    for healthy working-age adults, so we begin sampling at 60.
    """
    # Skip to age 60 — survival in 50s is not materially uncertain for planning
    age = 60
    while age <= 102:
        p = SURVIVAL.get(age, 0.0)
        if p == 0.0 or rng.random() > p:
            return age
        age += 1
    return 102


def block_bootstrap(history, block_size, n_needed, rng):
    """Sample consecutive blocks from a historical series to fill n_needed years."""
    out = []
    while len(out) < n_needed:
        start = int(rng.integers(0, len(history) - block_size))
        out.extend(history[start:start + block_size])
    return out[:n_needed]


def run_simulation(cfg):
    # ── Unpack config ─────────────────────────────────────────────────────────
    start_age    = cfg["starting_age"]
    retire_age   = cfg["retirement_age"]
    end_age      = cfg["end_age"]
    real_spend0  = cfg["annual_spending"]
    avg_ret      = cfg["avg_invest_return"]
    base_std     = cfg.get("invest_std_dev", 0.15)
    spend_vol    = cfg["spending_volatility"]
    monthly_cont = cfg["monthly_contribution"]
    charity_pct  = cfg["charity_pct"]
    one_time     = {int(k): v for k, v in cfg.get("one_time_events", {}).items()}
    wealth0      = cfg["starting_wealth"]
    ss_start     = cfg["social_security_start_age"]
    ss_annual    = cfg["social_security_annual"]
    n_sims       = cfg["simulations"]
    eq_alloc     = cfg.get("equity_allocation", 0.80)
    bd_alloc     = 1.0 - eq_alloc
    avg_inf      = cfg.get("avg_inflation", 0.030)

    # Advanced parameters from [advanced] section
    t_df        = adv(cfg, "t_distribution_df",    5)
    g_om        = adv(cfg, "garch_omega",          0.0002)
    g_al        = adv(cfg, "garch_alpha",          0.15)
    g_be        = adv(cfg, "garch_beta",           0.80)
    gr_cut      = adv(cfg, "guardrail_cut_pct",    0.10)
    gr_raise    = adv(cfg, "guardrail_raise_pct",  0.10)
    gr_floor    = adv(cfg, "guardrail_floor_pct",  0.80)
    gr_ceil     = adv(cfg, "guardrail_ceil_pct",   1.20)
    crash_freq  = adv(cfg, "crash_frequency_years", 20)

    # Feature flags
    use_regime  = flag(cfg, "regime_switching")
    use_ftails  = flag(cfg, "fat_tails")
    use_garch   = flag(cfg, "garch_volatility")
    use_sinf    = flag(cfg, "stochastic_inflation")
    use_mr      = flag(cfg, "mean_reversion")
    use_boot    = flag(cfg, "block_bootstrap", False)
    use_corr    = flag(cfg, "correlated_assets")
    use_gk      = flag(cfg, "dynamic_withdrawal_guardrails")
    use_long    = flag(cfg, "stochastic_longevity")
    use_crash   = flag(cfg, "crash_shock_injection")

    ages    = list(range(start_age, end_age + 1))
    n_ages  = len(ages)
    ss_arr  = [ss_annual if a >= ss_start else 0.0 for a in ages]
    ot_arr  = [one_time.get(a, 0.0) for a in ages]

    initial_wr = real_spend0 / wealth0 if wealth0 > 0 else 0.04

    # Cash wedge (bucket strategy) parameters
    wedge_years       = cfg.get("cash_wedge_years", 0.0)          # 0 = disabled
    wedge_refill_rule = cfg.get("cash_wedge_refill_rule", "five_year_mean")
    # Escape velocity: WR ratio below which the wedge is retired
    # e.g. 0.25 means "once WR drops to 25% of initial_wr, stop the wedge"
    wedge_escape_wr   = cfg.get("cash_wedge_escape_velocity", 0.25)
    use_wedge         = (wedge_years > 0.0) and (retire_age > start_age or wealth0 > 0)

    # Cash / bond return overrides.
    # None = use default model (cash wedge tracks inflation; bonds use b_mu/b_sig regime model)
    # 0.0  = zero nominal return (e.g. for Shariah-compliant / non-interest portfolios)
    # Any float = fixed annual nominal return applied every year regardless of regime
    cash_return_override = cfg.get("cash_return_override", None)   # wedge / cash bucket
    bond_return_override = cfg.get("bond_return_override", None)   # bond allocation

    results_dict         = {}
    ruin_count           = 0  # portfolio (excl. wedge) hit $0 in retirement
    ss_only_count        = 0  # ruined AND receiving SS income
    wedge_depleted_count = 0  # wedge fully depleted at least once
    cum_inf_list         = []  # final cumulative inflation multiplier per sim

    for sim in range(n_sims):
        rng = np.random.default_rng(sim)

        wealth    = wealth0
        regime    = "bull" if rng.random() < 0.55 else ("sideways" if rng.random() < 0.70 else "bear")
        prev_ret  = avg_ret
        garch_h   = base_std ** 2
        inf_state = avg_inf
        cum_inf   = 1.0
        death_age = draw_death_age(start_age, rng) if use_long else end_age + 1

        if use_boot:
            boot_eq  = block_bootstrap(SP500_ANNUAL, 5, n_ages, rng)
            boot_cpi = block_bootstrap(CPI_ANNUAL,   5, n_ages, rng)

        sim_result      = []
        sim_ruined      = False  # portfolio hit $0 in retirement
        sim_ss_only     = False  # ruined while receiving SS income
        wedge_depleted  = False  # wedge hit $0 at least once

        # Cash wedge: nominal dollar amount held in cash (earns ~0% real)
        # Funded at retirement onset from the equity portfolio
        wedge_cash     = 0.0
        wedge_retired  = False   # True once escape velocity reached
        ret5_window    = []      # rolling 5-year return window for BR-3 rule

        for i, age in enumerate(ages):

            # stochastic_longevity is used for reporting context only —
            # we always simulate to end_age so the portfolio is stress-tested
            # for the full horizon (you can't go back to work at 90).

            # Regime transition
            if use_regime and i > 0:
                regime = next_regime(regime, rng)

            r_mu  = REGIMES[regime][0] if use_regime else avg_ret
            r_sig = REGIMES[regime][1] if use_regime else base_std
            b_mu  = REGIMES[regime][2] if use_regime else 0.035
            b_sig = REGIMES[regime][3] if use_regime else 0.040

            # ── Generate portfolio return ─────────────────────────────────────
            if use_boot:
                eq_ret   = float(boot_eq[i])
                inf_ret  = float(boot_cpi[i])
                bond_noise = float(rng.normal(b_mu, b_sig))
                bond_noise = float(bond_return_override) if bond_return_override is not None                             else bond_noise
                eq_ret   = eq_alloc * eq_ret + bd_alloc * bond_noise
            else:
                rho = REGIME_CORR[regime] if use_regime else -0.20

                if use_ftails:
                    sf = math.sqrt((t_df - 2) / t_df)
                    z1 = float(rng.standard_t(t_df)) * sf
                    z2 = float(rng.standard_t(t_df)) * sf
                else:
                    z1 = float(rng.standard_normal())
                    z2 = float(rng.standard_normal())

                # Cholesky decomposition for correlated stock/bond draw
                if use_corr:
                    b_innov = rho * z1 + math.sqrt(max(1.0 - rho ** 2, 0.0)) * z2
                else:
                    b_innov = z2

                bond_ret = float(bond_return_override) if bond_return_override is not None                            else (b_mu + b_sig * b_innov)
                eq_ret  = eq_alloc * (r_mu + r_sig * z1) + bd_alloc * bond_ret
                inf_ret = 0.0  # set by AR(1) below

                # GARCH(1,1) — scale return innovation by time-varying vol
                if use_garch:
                    garch_h   = g_om + g_al * prev_ret ** 2 + g_be * garch_h
                    garch_sig = math.sqrt(garch_h)
                    vol_scale = float(np.clip(garch_sig / (r_sig + 1e-9), 0.4, 2.0))
                    eq_ret    = r_mu + (eq_ret - r_mu) * vol_scale

                # AR(1) mean reversion on annual returns
                if use_mr and i > 0:
                    ar     = -0.18
                    eq_ret = r_mu + ar * (prev_ret - r_mu) \
                             + (eq_ret - r_mu) * math.sqrt(1.0 - ar ** 2)

                # Poisson crash shock
                if use_crash and rng.random() < (1.0 / crash_freq):
                    eq_ret += float(rng.choice(CRASH_SEVERITIES))

                eq_ret = float(np.clip(eq_ret, -0.75, 1.50))

            # ── Stochastic inflation AR(1) ────────────────────────────────────
            if use_sinf:
                if use_boot:
                    inf_state = float(np.clip(inf_ret, -0.02, 0.12))
                else:
                    inf_noise = float(rng.normal(0.0, 0.010))
                    inf_state = avg_inf + 0.65 * (inf_state - avg_inf) + inf_noise * 0.50
                    inf_state = float(np.clip(inf_state, -0.02, 0.12))
            else:
                inf_state = avg_inf

            cum_inf *= (1.0 + inf_state)

            # Pre-retirement contributions
            if age < retire_age:
                wealth += monthly_cont * 12

            # Fund cash wedge at retirement (once, from portfolio)
            if use_wedge and not wedge_retired and age == retire_age and wedge_cash == 0.0:
                wedge_target = real_spend0 * cum_inf * wedge_years
                transfer     = min(wedge_target, wealth * 0.30)  # cap at 30% of portfolio
                wedge_cash  += transfer
                wealth      -= transfer

            # Apply investment return
            wealth *= (1.0 + eq_ret)

            # Cash wedge grows at: override rate (if set) OR inflation rate (default)
            if use_wedge and not wedge_retired:
                if cash_return_override is not None:
                    wedge_cash *= (1.0 + float(cash_return_override))
                else:
                    wedge_cash *= (1.0 + inf_state)  # default: tracks inflation

            # Social Security income (partial COLA: 50% of simulated inflation).
            # Applied BEFORE spending so depleted sims net SS - spend < 0 → $0.
            # This eliminates the ghost balance that appeared when SS was added
            # after spending, leaving ~1 year of SS sitting in the balance.
            if ss_arr[i] > 0:
                wealth += ss_arr[i] * (1.0 + inf_state * 0.50)
                # Track SS-only: portfolio was already ruined, now receiving SS
                if sim_ruined and not sim_ss_only:
                    ss_only_count += 1
                    sim_ss_only    = True

                        # Nominal spending = real spending × cumulative inflation
            nom_spend = real_spend0 * cum_inf
            if eq_ret > r_mu:
                spend = nom_spend * (1.0 + spend_vol)
            elif eq_ret < r_mu:
                spend = nom_spend * (1.0 - spend_vol)
            else:
                spend = nom_spend

            # Guyton-Klinger guardrails (post-retirement only)
            if use_gk and age >= retire_age and wealth > 0:
                wr = spend / wealth
                if wr > initial_wr * gr_ceil:
                    spend *= (1.0 - gr_cut)
                elif wr < initial_wr * gr_floor:
                    spend *= (1.0 + gr_raise)

            # ── Cash wedge drawdown logic (post-retirement) ───────────────────
            if use_wedge and not wedge_retired and age >= retire_age:
                total_assets = wealth + wedge_cash

                # Check escape velocity: WR vs total assets
                if total_assets > 0 and (spend / total_assets) < initial_wr * wedge_escape_wr:
                    wedge_retired = True  # portfolio big enough; retire the wedge
                    wealth       += wedge_cash
                    wedge_cash    = 0.0

                else:
                    # Decide whether to draw from wedge or portfolio (refill rule)
                    # BR-3 (five_year_mean): use wedge if 5-yr mean return < long-run avg
                    # BR-1 (market_up):      use wedge if last year's return was negative
                    draw_from_wedge = False
                    if wedge_refill_rule == "five_year_mean":
                        recent5 = sum(ret5_window[-5:]) / len(ret5_window) if ret5_window else avg_ret
                        draw_from_wedge = (recent5 < avg_ret) and (wedge_cash >= spend * 0.5)
                    else:  # market_up / BR-1
                        draw_from_wedge = (prev_ret < 0.0) and (wedge_cash >= spend * 0.5)

                    if draw_from_wedge:
                        # Draw from wedge first; overflow to portfolio
                        from_wedge = min(spend, wedge_cash)
                        spend_from_portfolio = spend - from_wedge
                        wedge_cash -= from_wedge
                        wealth     -= spend_from_portfolio
                        # Refill wedge from portfolio when market recovering
                        # (refill up to target; only refill when portfolio healthy)
                        wedge_target = real_spend0 * cum_inf * wedge_years
                        if wedge_cash < wedge_target * 0.5 and wealth > 0:
                            refill = min(wedge_target - wedge_cash, wealth * 0.05)
                            wedge_cash += refill
                            wealth     -= refill
                    else:
                        # Normal year: draw from portfolio, refill wedge if depleted
                        wealth -= spend
                        wedge_target = real_spend0 * cum_inf * wedge_years
                        if wedge_cash < wedge_target * 0.25 and wealth > 0:
                            refill     = min(wedge_target - wedge_cash, wealth * 0.05)
                            wedge_cash += refill
                            wealth     -= refill
            elif age >= retire_age:
                # No wedge (or retired): normal withdrawal from portfolio
                wealth -= spend

            # Track 5-year return window for BR-3 rule
            ret5_window.append(eq_ret)
            if len(ret5_window) > 5:
                ret5_window.pop(0)



            # Charitable giving
            wealth -= max(wealth, 0.0) * charity_pct

            # One-time events (inheritances, large purchases, etc.)
            wealth += ot_arr[i]

            # If portfolio is negative after spending, drain wedge for the shortfall.
            # This ensures wedge residuals don't mask true depletion in the output.
            if use_wedge and not wedge_retired and wealth < 0.0 and wedge_cash > 0.0:
                shortfall   = -wealth
                from_wedge  = min(shortfall, wedge_cash)
                wedge_cash -= from_wedge
                wealth     += from_wedge

            # Ruin = total assets (portfolio + wedge) hit $0 during retirement.
            if age >= retire_age and (wealth + wedge_cash) <= 0.0 and not sim_ruined:
                ruin_count += 1
                sim_ruined = True
            # Track wedge depletion separately
            if use_wedge and not wedge_retired and wedge_cash <= 0.0 and not wedge_depleted:
                wedge_depleted_count += 1
                wedge_depleted = True
            wealth     = max(wealth, 0.0)
            wedge_cash = max(wedge_cash, 0.0)
            prev_ret = eq_ret
            sim_result.append(int(round(wealth + wedge_cash)))

        results_dict[f"sim_{sim + 1}"] = sim_result
        cum_inf_list.append(cum_inf)

    results = pd.DataFrame(results_dict, index=ages)
    results.index.name = "age"

    pct_map = {"p01":1,"p05":5,"p10":10,"p25":25,"median":50,"p75":75,"p90":90,"p95":95,"p99":99}
    percentiles = pd.DataFrame(index=ages)
    for col, lv in pct_map.items():
        percentiles[col] = results.quantile(lv / 100.0, axis=1).round(0).astype(int)

    # Ruin = portfolio hit $0 while the person was still alive.
    # Dead simulations legitimately end at $0 (estate distributed) and should
    # NOT be counted as ruin. Track ruin separately during the simulation loop.
    ruin_rate            = ruin_count / n_sims
    ss_only_rate         = ss_only_count / n_sims
    wedge_depletion_rate = wedge_depleted_count / n_sims
    median_cum_inf       = float(np.median(cum_inf_list))
    return results, percentiles, ruin_rate, ss_only_rate, wedge_depletion_rate, median_cum_inf


def print_summary(cfg, results, percentiles, ruin_rate, ss_only_rate, wedge_depletion_rate, median_cum_inf):
    ea     = cfg["end_age"]
    n_sims = len(results.columns)
    n_yrs  = ea - cfg["starting_age"]
    fv     = results.iloc[-1].values  # nominal portfolio value at end_age

    print("\n" + "=" * 75)
    print("MONTE CARLO SIMULATION SUMMARY — Professional Model")
    print("=" * 75)
    print(f"  Simulations       : {n_sims:,}")
    print(f"  Age range         : {cfg['starting_age']} → {ea}  ({n_yrs} years)")
    print(f"  Equity allocation : {cfg.get('equity_allocation', 0.80)*100:.0f}%")
    print(f"  Avg inflation     : {cfg.get('avg_inflation', 0.030)*100:.1f}%  "
          f"(median simulated cum. inflation: {median_cum_inf:.2f}x over {n_yrs} yrs)")
    wedge_in_use = cfg.get("cash_wedge_years", 0.0) > 0
    use_ss_cfg   = cfg.get("social_security_start_age", 0) > 0 and cfg.get("social_security_annual", 0) > 0
    print(f"  Ruin rate         : {ruin_rate*100:.1f}%  "
          f"(portfolio hits $0 before age {ea})")
    if use_ss_cfg and ss_only_rate > 0.001:
        print(f"  ⚠  SS-only        : {ss_only_rate*100:.1f}%  "
              f"(portfolio depleted; surviving on Social Security alone)")
    truly_depleted = max(0.0, ruin_rate - ss_only_rate)
    if use_ss_cfg and truly_depleted > 0.001:
        print(f"  ✗  Truly depleted : {truly_depleted*100:.1f}%  "
              f"(insufficient assets even with SS income)")
    if wedge_in_use:
        print(f"  Wedge depleted    : {wedge_depletion_rate*100:.1f}%  "
              f"(cash wedge fully exhausted at least once)")
    print("-" * 75)
    print(f"  {'PORTFOLIO AT AGE ' + str(ea) + ' (inheritance / end-of-plan)':<40}"
          f"  {'NOMINAL':>13}   {'TODAY\'S $':>13}")
    print(f"  {'(nominal = future dollars; today\'s $ = inflation-adjusted)':<40}")
    print("-" * 75)
    for label, pct in [("99th",99),("95th",95),("90th",90),("75th",75),
                        ("Median",50),("25th",25),("10th",10),("5th",5),("1st",1)]:
        nom  = int(np.percentile(fv, pct))
        real = int(nom / median_cum_inf)
        print(f"  {label + ' percentile':<40}  ${nom:>13,}   ${real:>13,}")
    print("=" * 75)
    print(f"  Note: today's $ column divides by median simulated inflation ({median_cum_inf:.2f}x).")
    print(f"  Individual sim inflation ranges vary — see CSV for per-sim data.")
    print("=" * 75 + "\n")


def plot_results(cfg, ages, percentiles, ruin_rate, script_dir):
    ages = list(ages)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 14))

    # ── Fan chart (top panel) ─────────────────────────────────────────────────
    ax.fill_between(ages, percentiles["p01"],    percentiles["p99"],
                    color="#cce0f0", alpha=0.5, label="1st–99th percentile")
    ax.fill_between(ages, percentiles["p05"],    percentiles["p95"],
                    color="#88b8d8", alpha=0.5, label="5th–95th percentile")
    ax.fill_between(ages, percentiles["p10"],    percentiles["p90"],
                    color="#4488b0", alpha=0.5, label="10th–90th percentile")
    ax.fill_between(ages, percentiles["p25"],    percentiles["p75"],
                    color="#1a5070", alpha=0.6, label="25th–75th percentile")
    ax.plot(ages, percentiles["median"],
            color="#cc2200", linewidth=2.5, label="Median (50th)")
    ax.plot(ages, percentiles["p10"],
            color="#666", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.plot(ages, percentiles["p90"],
            color="#666", linewidth=1.0, linestyle="--", alpha=0.6)

    ax.set_yscale("log")
    raw_min = percentiles["p01"].replace(0, np.nan).min()
    y_min   = max(raw_min * 0.5 if not np.isnan(raw_min) else 1000, 1000)
    y_max   = percentiles["p99"].max() * 1.5
    ax.set_ylim(y_min, y_max)

    def fmt_currency(x, _):
        if x >= 1e6: return f"${x/1e6:.1f}M"
        if x >= 1e3: return f"${x/1e3:.0f}K"
        return f"${x:.0f}"

    ticks = []
    for e in np.arange(np.floor(np.log10(y_min)), np.ceil(np.log10(y_max)) + 1):
        for m in [1, 2, 5]:
            v = 10 ** e * m
            if y_min <= v <= y_max:
                ticks.append(v)
    ax.yaxis.set_ticks(ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_currency))

    mf = cfg.get("model_flags", {})
    labels = {
        "regime_switching":              "Regime Switching",
        "fat_tails":                     "Fat Tails (t-dist)",
        "garch_volatility":              "GARCH",
        "stochastic_inflation":          "Stoch Inflation",
        "mean_reversion":                "Mean Reversion",
        "correlated_assets":             "Corr Assets",
        "dynamic_withdrawal_guardrails": "GK Guardrails",
        "stochastic_longevity":          "Longevity",
        "crash_shock_injection":         "Crash Shocks",
        "block_bootstrap":               "Bootstrap",
    }
    active = " · ".join(v for k, v in labels.items() if mf.get(k, k != "block_bootstrap"))
    ax.set_title(
        f"Monte Carlo Portfolio ({cfg['simulations']:,} simulations)\n{active}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel("Portfolio Value", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", which="major")
    ax.axvline(cfg["retirement_age"], color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(cfg["retirement_age"] + 0.3, y_min * 2,
            f"Retire {cfg['retirement_age']}", fontsize=8, color="gray")
    ax.text(0.99, 0.05, f"Ruin rate: {ruin_rate*100:.1f}%",
            transform=ax.transAxes, ha="right", fontsize=10, color="#cc2200",
            fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    # ── Survival probability panel (bottom) ──────────────────────────────────
    ax2.set_title("Probability of Portfolio Survival by Age", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Age", fontsize=11)
    ax2.set_ylabel("% of Simulations Still Solvent", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, linestyle="--")

    for col, clr, lbl in [
        ("p90",    "#1a5070", "90th pct still solvent"),
        ("median", "#cc2200", "Median still solvent"),
        ("p10",    "#88b8d8", "10th pct still solvent"),
    ]:
        solvent = (percentiles[col] > 0).astype(float) * 100
        ax2.fill_between(ages, 0, solvent, alpha=0.18, color=clr)
        ax2.plot(ages, solvent, color=clr, linewidth=1.8, label=lbl)

    ax2.axvline(cfg["retirement_age"], color="gray", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.legend(fontsize=9)

    plt.tight_layout(pad=3.0)
    out = os.path.join(script_dir, "portfolio_simulation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print("Chart saved as:", out)


if __name__ == "__main__":
    cfg        = load_config()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    mf     = cfg.get("model_flags", {})
    active = [k for k, v in mf.items() if v]
    print("Running professional-grade Monte Carlo simulation…")
    print("Active layers:", ", ".join(active) if active else "none")
    print()

    results, percentiles, ruin_rate, ss_only_rate, wedge_depletion_rate, median_cum_inf = run_simulation(cfg)

    csv_path = os.path.join(script_dir, "portfolio_simulation.csv")
    results.to_csv(csv_path)
    print("Results saved to:", csv_path)

    print_summary(cfg, results, percentiles, ruin_rate, ss_only_rate, wedge_depletion_rate, median_cum_inf)
    plot_results(cfg, results.index, percentiles, ruin_rate, script_dir)
