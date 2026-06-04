# Monte Carlo Retirement Simulator

A retirement portfolio simulator built on a **global equity historical block bootstrap** engine. Each simulation path is constructed by stitching together random blocks of real MSCI World annual returns and CPI inflation, then calibrating the path to a forward-looking return assumption. This preserves the actual shape of market history — crash depth, volatility clustering, lost decades — while discounting the unrepeatable US-superpower premium embedded in raw historical data.

No regime models. No synthetic returns. No GARCH. No fat-tail generators. The only randomness is which historical blocks get stitched together for each simulated lifetime.

## Quick start

```bash
cp config.example.toml config.toml   # first time only
# edit config.toml for your situation
./run_simulation.sh
```

Outputs:
- `portfolio_simulation.csv` — per-simulation wealth by age (one column per sim)
- `portfolio_percentiles.csv` — p01/p05/p10/p25/median/p75/p90/p95/p99 by age
- Summary table printed to stdout

`run_simulation.sh` uses your system `python3`. No external dependencies — standard library only (Python 3.11+ required for `tomllib`).

## How the engine works

1. **Block bootstrap** — a simulation path is built by randomly drawing variable-length slices (5, 10, 15, or 20 years) from the MSCI World annual return array and the CPI array. Slices are concatenated until the full horizon (`end_age - starting_age + 1` years) is covered. Equity returns and inflation are always drawn from the *same* historical slice, so their co-movement is preserved.

2. **CAGR calibration** — the raw bootstrapped equity series has a geometric mean of ~9.7% (MSCI World 1970–2023). Each path is shifted in log-space so its geometric mean equals `target_equity_cagr`. Crash magnitude and volatility profile are preserved; only the long-run drift is adjusted. The same shift is applied to inflation toward `target_inflation`.

3. **Path simulation** — the calibrated return and inflation series drive a year-by-year engine covering contributions, portfolio growth, inflation-indexed spending, Social Security, a cash wedge, charity, and one-time events.

### Why MSCI World instead of S&P 500?

The S&P 500 1928–2023 geometric mean is ~9.7% nominal. The MSCI World 1970–2023 geometric mean is also ~9.7% nominal — the same issue. Both reflect the same post-WWII developed-market bull run. The Dimson–Marsh–Staunton Global Investment Returns Yearbook (2025) estimates a forward-looking global DM equity return of roughly 6–7% nominal, discounting the structural tailwinds that cannot repeat. Setting `target_equity_cagr = 0.07` applies that discount explicitly, while still using MSCI World's real crash history (1974: −25%, 2002: −20%, 2008: −40%) as the distribution of outcomes.

## Configuration

All configuration lives in `config.toml`. Copy `config.example.toml` as a starting point.

### Human levers

These are the knobs you actually care about:

| Parameter | Description |
|---|---|
| `starting_age` | Your current age |
| `retirement_age` | Age you stop contributing and start drawing |
| `end_age` | Simulation end age (e.g. 100) |
| `starting_wealth` | Current investable net worth |
| `monthly_contribution` | Monthly savings during accumulation phase |
| `annual_spending` | Target annual spending in today's dollars |
| `spending_volatility` | Fraction to flex spending up/down in good/bad equity years (e.g. `0.10` = ±10%) |
| `charity_pct` | Annual charitable giving as fraction of portfolio balance |
| `equity_allocation` | Fraction in global equities; remainder earns `bond_return_override` |
| `social_security_start_age` | Age at which SS income begins |
| `social_security_annual` | Annual SS benefit in today's dollars |
| `simulations` | Number of Monte Carlo paths (10,000 recommended) |

### Return assumptions

| Parameter | Default | Description |
|---|---|---|
| `target_equity_cagr` | `0.07` | Forward-looking nominal equity CAGR (~4% real at 3% inflation). DMS Yearbook 2025 estimate for global DM. Lower = more conservative. Set to `0.097` to use raw MSCI history unmodified. |
| `target_inflation` | `0.03` | Nominal CPI target. 3% is modestly above the Fed 2% anchor. |

### Cash wedge

The cash wedge holds a liquid cash buffer at retirement to avoid selling equities in down years.

| Parameter | Description |
|---|---|
| `cash_wedge_years` | Years of inflation-adjusted spending held in cash. Set to `0` to disable. |
| `cash_wedge_refill_rule` | `"five_year_mean"` — use wedge when 5-year avg return is below target; `"negative_year"` — use in any down year |
| `cash_wedge_escape_velocity` | Dissolve the wedge permanently once withdrawal rate drops below this fraction of the initial retirement WR. E.g. `0.25` means once your WR is ¼ of what it was at retirement, the wedge is no longer needed. |
| `cash_return_override` | Nominal annual return on the cash bucket (default `0.0` — conservative) |
| `bond_return_override` | Nominal annual return for the non-equity allocation (default `0.0`) |

### Bootstrap tuning

```toml
bootstrap_block_sizes = [5, 10, 15, 20]  # block lengths in years
```

Longer blocks preserve more sequence autocorrelation (e.g. a whole lost decade stays together). Shorter blocks allow more recombination. The default mix of 5–20 year blocks is a reasonable balance.

### One-time events

```toml
[one_time_events]
"45" = 1000000    # positive = inflow (inheritance, equity vest, property sale)
"55" = -100000    # negative = outflow (large purchase, tax bill)
"60" = 250000
```

Keys are ages as quoted strings. Values are nominal dollars at the time of the event (not today's dollars).

## Understanding the output

```
 PORTFOLIO AT AGE 100                     NOMINAL         TODAY'S $
---------------------------------------------------------------------------
 99th percentile                      $46,077,966        $7,157,282
 ...
 Median percentile                    $35,925,408        $5,580,287
 ...
 1st percentile                       $20,920,986        $3,249,653
```

- **NOMINAL** — raw portfolio value at end age in future dollars
- **TODAY'S $** — nominal value deflated by the median cumulative inflation across all simulations. This is the purchasing-power equivalent in current dollars.
- **Depleted anytime** — fraction of simulations where the total portfolio (portfolio + wedge) hit zero at any point in retirement
- **Depleted at `end_age`** — fraction where the final year value was zero
- **SS-only anytime** — fraction where the portfolio was exhausted but Social Security income alone covered spending
- **Wedge depleted** — fraction where the cash wedge ran to zero before the escape-velocity condition triggered. A high number here (including 100%) is expected when your withdrawal rate is very low — the portfolio grows quickly enough that the wedge escape condition fires and the wedge is merged back into the portfolio.

## Design

The simulator is a single Python file with no external dependencies.

**Key functions:**

- `bootstrap_historical_series(cfg, rng)` — draws one bootstrapped + CAGR-shifted path
- `_shift_to_target(series, target_cagr)` — log-space CAGR calibration
- `simulate_path(cfg, seed)` — runs one full lifetime simulation, returns `SimulationPath`
- `apply_guardrail_cut(...)` — cuts spending if withdrawal rate exceeds the ceiling (raise side omitted — spending grows only via inflation indexing, preventing compounding spending in bull runs)
- `run_simulation(cfg)` — runs N paths and aggregates results
- `print_summary(...)` — prints the stdout summary table

**Data structures:**

- `SimulationConfig` — frozen dataclass, all config fields
- `YearState` — one year of simulation state (wealth, spending, SS, inflation, etc.)
- `SimulationPath` — full lifetime path plus depletion/SS flags

## Testing

Unit tests live in `tests/test_simulator.py`. Run:

```bash
python3 -m unittest discover -s tests -v
```

Tests cover:

- Bootstrap series length matches horizon exactly
- All bootstrapped values are drawn from the historical arrays (not synthesized)
- Determinism: same seed → same path
- Earlier retirement produces lower terminal wealth (all else equal)
- High spending triggers depletion; low spending does not
- Social Security income compounds with inflation correctly