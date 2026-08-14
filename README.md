# Monte Carlo Retirement Simulator

A retirement portfolio simulator built on a **global equity historical block bootstrap** engine. Each simulation path is constructed by stitching together random blocks of real MSCI World annual returns and CPI inflation, then calibrating the path to a forward-looking return assumption. This preserves the actual shape of market history — crash depth, volatility clustering, lost decades, etc.

## Quick start

```bash
cp config.example.toml config.toml   # first time only
# edit config.toml for your situation
./run_simulation.sh
```

Outputs:
- `portfolio_simulation.csv` — per-simulation nominal wealth by age
- `portfolio_percentiles.csv` — nominal percentiles by age
- `portfolio_percentiles_real.csv` — today's-dollar percentiles (each path deflated by **that path's** inflation)
- Summary table printed to stdout

`run_simulation.sh` uses your system `python3`. No external dependencies — standard library only (Python 3.11+ required for `tomllib`).

## How the engine works

1. **Block bootstrap** — a path is built by randomly drawing short slices (default 4, 5, 6, or 8 years) from the MSCI World annual return array and the CPI array. Equity and inflation always come from the same historical slice.
2. **CAGR calibration** — each raw series is shifted so shocks stay relative to the global historical mean, while the long-run drift matches `target_equity_cagr` / `target_inflation`.
3. **Path simulation** — contributions, growth, inflation-indexed spending, Social Security, optional pre-Medicare extra, optional withdrawal tax, cash wedge, charity, and one-time events.

## Configuration

All configuration lives in `config.toml`. Copy `config.example.toml` as a starting point.

### Human levers

| Parameter | Description |
|---|---|
| `starting_age` | Your current age |
| `retirement_age` | Age you stop contributing and start drawing |
| `end_age` | Simulation end age (e.g. 100) |
| `starting_wealth` | Current investable net worth |
| `monthly_contribution` | Monthly savings during accumulation |
| `inflate_contributions` | `true` (default) grows contributions with inflation |
| `annual_spending` | Core lifestyle spending in today's dollars |
| `pre_medicare_extra_annual` | Extra today's-$ spend while retired and younger than `medicare_age` |
| `medicare_age` | Age at which the extra stops (default 65) |
| `withdrawal_tax_rate` | Tax applied to portfolio withdrawals after SS (see below) |
| `spending_volatility` | Fraction to flex core spending in good/bad equity years |
| `equity_allocation` | Fraction in global equities; remainder needs `bond_return_override` |
| `social_security_start_age` | Age at which SS income begins |
| `social_security_annual` | Annual SS benefit in today's dollars for the claiming age you chose |
| `simulations` | Number of Monte Carlo paths (10,000 recommended) |

### Withdrawal tax

`withdrawal_tax_rate` is a single blended rate on the portfolio draw **after** Social Security has already offset spending.

- Set it to `0.0` if taxes are already inside `annual_spending`.
- Set it to `0.10`–`0.15` if `annual_spending` is a lifestyle number and most withdrawals will come from pretax accounts. That is the usual case.
- There is no bracket engine, Roth/pretax split, or capital-gains vs ordinary-income split. If your tax picture is unusual, leave the rate at 0 and put tax in the spending line yourself.

### Why there is no expense-ratio field

Fund fees are easier to fold into `target_equity_cagr` than to capture with one number. A three-fund plus stock portfolio does not have a single published ER. If your all-in cost is about 0.15%, use `target_equity_cagr = 0.0685` instead of `0.07`.

### Return assumptions

| Parameter | Default | Description |
|---|---|---|
| `target_equity_cagr` | `0.07` | Forward-looking nominal equity CAGR. Lower = more conservative. |
| `target_inflation` | `0.03` | Nominal CPI target. |

### Cash wedge

| Parameter | Description |
|---|---|
| `cash_wedge_years` | Years of inflation-adjusted core spending held in cash. `0` disables. |
| `cash_wedge_refill_rule` | `five_year_mean` or `negative_year` |
| `cash_wedge_escape_velocity` | Dissolve the wedge once WR drops below this fraction of the initial WR |

### Bootstrap tuning

```toml
bootstrap_block_sizes = [4, 5, 6, 8]
```

Short blocks keep multi-year crashes together but recombine history more freely. 30–40 year blocks drawn from only 54 years of data are almost the same handful of overlapping lifetimes.

### One-time events

```toml
[one_time_events]
"45" = 1000000    # today's dollars; inflated to that age
"55" = -100000
```

## Understanding the output

```
 PORTFOLIO AT AGE 100                     NOMINAL         TODAY'S $
---------------------------------------------------------------------------
 99th percentile                      $46,077,966        $7,157,282
 ...
```

- **NOMINAL** — future dollars at end age
- **TODAY'S $** — each simulation is deflated by **that path's** cumulative inflation, then the percentile is taken. This is not “nominal percentile ÷ median inflation.”
- **Depleted anytime** — fraction of paths where portfolio + wedge hit zero at any retirement year
- **Depleted at `end_age`** — fraction whose final recorded year was zero
- **Wedge depleted** — wedge was never retired and is empty at the end

Spending guardrails are always on: if the withdrawal rate exceeds 120% of the first retirement year's rate, core lifestyle spending is cut 10% permanently (healthcare extra is not cut). Depletion rates therefore assume you will accept those cuts. They are not “I kept spending $X forever.”

## Testing

```bash
./run_tests.sh
```

Or:

```bash
python3 -m unittest -v test_monte_carlo_portfolio.py
```
