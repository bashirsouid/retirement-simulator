# Monte Carlo Investment Portfolio Simulator

A Python-based Monte Carlo simulation tool for modeling investment portfolio growth and decay over time. This script helps you understand the range of possible retirement outcomes based on your financial parameters and market variability.

## ⚠️ Disclaimer

**This tool is AI-generated and untested. Use at your own risk.** This simulator is provided for educational and planning purposes only. It is not financial advice, and you should consult with a qualified financial advisor before making any major financial decisions. The default configuration values are roughly based on research of national averages in the USA but are not completely verified (AI was used to research these values). Always validate assumptions with current data and professional guidance.

## Features

- **1000-iteration Monte Carlo simulations** (configurable) of portfolio outcomes
- **Realistic investment returns** using normal distribution (bell curve) with configurable standard deviation
- **Dynamic spending** that adjusts based on investment performance
- **Monthly contributions** until retirement
- **Social Security income** starting at a specified age
- **One-time events** (inheritances, home sales, major expenses, etc.)
- **Annual charitable donations** as a percentage of net worth
- **CSV output** with all simulation iterations
- **Percentile visualization** showing outcome ranges (1st, 5th, 10th, 50th, 90th, 95th, 99th percentiles)
- **Console summary** with key statistics

## Prerequisites

- Python 3.7+
- Bash shell (Linux, macOS, or WSL on Windows)

## Installation & Usage

### Quick Start

1. **Clone or download the repository**

2. **Make the script executable:**
   ```bash
   chmod +x run_simulation.sh
   ```

3. **Run the simulation:**
   ```bash
   ./run_simulation.sh
   ```

The script will:
- Create a Python virtual environment (first run only)
- Install required dependencies (numpy, pandas, matplotlib)
- Copy `config.example.json` to `config.json` (if `config.json` doesn't exist)
- Run the simulation
- Generate `portfolio_simulation.csv` and `portfolio_simulation.png`

### Customizing Your Scenario

Edit `config.json` to adjust parameters for your specific situation. The bash script will preserve your `config.json` on subsequent runs.

**Example: Create different scenarios**
```bash
cp config.json config_conservative.json
# Edit config_conservative.json with lower returns
cp config_conservative.json config.json
./run_simulation.sh
```

## Configuration Parameters

All parameters are defined in `config.json`:

### Age & Timeline
- **`starting_age`** (int): Your current age. Default: 40
- **`retirement_age`** (int): Age at which you stop making contributions and begin living off savings. Default: 62
- **`end_age`** (int): Age to simulate until. Default: 90

### Spending
- **`annual_spending`** (float): Annual expenses in today's dollars. Default: $60,000
- **`spending_volatility`** (float): Fractional adjustment to spending in good/bad investment years. For example, 0.05 means spending increases 5% in good years and decreases 5% in bad years. Default: 0.05

### Investment Returns
- **`avg_invest_return`** (float): Expected average annual investment return (e.g., 0.05 = 5%). Default: 0.05
- **`invest_std_dev`** (float): Standard deviation of returns (volatility). Controls how much returns vary year-to-year. Higher values = more variable returns. Default: 0.12 (12%)
  - Conservative portfolio (bonds + stocks): 0.08–0.10
  - Balanced portfolio (60/40 stocks/bonds): 0.10–0.12
  - Aggressive portfolio (mostly stocks): 0.15–0.18

### Contributions & Withdrawals
- **`monthly_contribution`** (float): Monthly savings added to portfolio (stops at retirement). Default: $1,000
- **`charity_pct`** (float): Annual charitable donation as a percentage of net worth. For example, 0.025 = 2.5% annually. Default: 0.01 (1%)

### Windfalls & Major Events
- **`one_time_events`** (object): One-time income (inheritance, home sale) or expenses (major purchase). Keys are ages, values are dollar amounts (positive for income, negative for expenses). Default: $100,000 at age 50
  - Example with multiple events:
    ```json
    "one_time_events": {
      "50": 100000,
      "65": -50000,
      "75": 250000
    }
    ```

### Starting Conditions
- **`starting_wealth`** (float): Initial portfolio value. Default: $45,000

### Social Security
- **`social_security_start_age`** (int): Age at which Social Security payments begin. Default: 62
- **`social_security_annual`** (float): Annual Social Security benefit amount in today's dollars. Default: $32,000
  - Note: The maximum Social Security benefit for a high-earner claiming at 62 in 2025 is roughly $2,660/month (~$31,920/year), but this varies by earning history and claiming age.

### Simulation Settings
- **`simulations`** (int): Number of Monte Carlo iterations to run. More iterations = smoother results but longer runtime. Default: 1000

## Understanding the Output

### Console Output

```
==============================================================
MONTE CARLO SIMULATION SUMMARY
==============================================================

Number of simulations: 1000
Age range: 40 to 90
Average return: 5.0% per year
Standard deviation: 12.0% per year

--------------------------------------------------------------
FINAL PORTFOLIO VALUE STATISTICS (at age 90)
--------------------------------------------------------------
99th percentile: $1,234,567
95th percentile: $  987,654
90th percentile: $  876,543
Median (50th):   $  654,321
10th percentile: $  432,109
5th percentile:  $  321,098
1st percentile:  $  123,456
==============================================================
```

This shows the distribution of final portfolio values across all 1000 simulations, helping you understand best-case, worst-case, and most-likely outcomes.

### CSV Output (`portfolio_simulation.csv`)

Contains one row per age and one column per simulation. You can import this into Excel, Pandas, or any spreadsheet tool for further analysis.

### Chart Output (`portfolio_simulation.png`)

A visualization showing:
- **Black line**: Median (50th percentile) outcome
- **Gray band (darkest)**: 10th–90th percentile range (most likely outcomes)
- **Gray band (medium)**: 5th–95th percentile range
- **Gray band (lightest)**: 1st–99th percentile range (extremes)

## Notes on Assumptions

- **Inflation ignored**: All values are in "today's dollars." Inflation is not modeled, keeping calculations simpler and focusing on real (inflation-adjusted) returns.
- **Normal distribution returns**: Investment returns follow a bell curve with occasional extreme years, mimicking historical market behavior.
- **Spending adjusts with performance**: In above-average return years, spending increases slightly; in below-average years, it decreases. This simulates adaptive spending behavior.
- **No taxes**: Tax implications are not included in the model.
- **No employer matching**: Contributions are assumed to be from personal savings, not employer matches.

## Example Scenarios

### Conservative (Low Risk)
```json
{
  "avg_invest_return": 0.04,
  "invest_std_dev": 0.08,
  "annual_spending": 50000,
  "monthly_contribution": 2000
}
```

### Balanced (Medium Risk)
```json
{
  "avg_invest_return": 0.06,
  "invest_std_dev": 0.12,
  "annual_spending": 60000,
  "monthly_contribution": 1500
}
```

### Aggressive (High Risk)
```json
{
  "avg_invest_return": 0.08,
  "invest_std_dev": 0.18,
  "annual_spending": 70000,
  "monthly_contribution": 1000
}
```

## Troubleshooting

**Error: "config.json not found"**
- Ensure `config.example.json` exists in the same directory as the scripts, or create `config.json` manually.

**Error: "Python 3 is not installed"**
- Install Python 3.7 or later from python.org or your system package manager.

**Script runs but produces no output**
- Check that matplotlib is installed correctly. The plot window may appear in the background on some systems.

**CSV values look wrong**
- Verify your configuration parameters. For example, very high spending relative to starting wealth will lead to depletion.

## Files

- `run_simulation.sh` – Bash script to set up environment and run simulation
- `monte_carlo_portfolio.py` – Main Python simulation script
- `config.example.json` – Example configuration file (commit to repo)
- `config.json` – Your personal configuration (generated from example, add to .gitignore)
- `portfolio_simulation.csv` – Output CSV with all simulation iterations
- `portfolio_simulation.png` – Output visualization chart
- `.gitignore` – Excludes personal config, outputs, and venv from version control

## License

This tool is provided as-is for educational and personal use.

## Contributing

Feel free to fork, modify, and improve this tool. Some potential enhancements:
- Tax modeling
- Inflation adjustment
- Asset allocation rebalancing
- Withdrawal sequencing optimization
- More sophisticated return distributions (log-normal, Student-t)
