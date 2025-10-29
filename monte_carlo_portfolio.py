import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os

# Load configuration from config.json
config_file = os.path.join(os.path.dirname(__file__), 'config.json')

if not os.path.exists(config_file):
    print(f"Error: config.json not found at {config_file}")
    exit(1)

with open(config_file, 'r') as f:
    config = json.load(f)

# Extract parameters from config
starting_age = config['starting_age']
retirement_age = config['retirement_age']
end_age = config['end_age']
annual_spending = config['annual_spending']
avg_invest_return = config['avg_invest_return']
invest_std_dev = config.get('invest_std_dev', 0.12)  # Standard deviation of returns (default 12%)
spending_volatility = config['spending_volatility']
monthly_contribution = config['monthly_contribution']
charity_pct = config['charity_pct']
one_time_events = {int(k): v for k, v in config['one_time_events'].items()}
starting_wealth = config['starting_wealth']
social_security_start_age = config['social_security_start_age']
social_security_annual = config['social_security_annual']
simulations = config['simulations']

np.random.seed(42)  # Reproducibility; comment to randomize each run

ages = np.arange(starting_age, end_age + 1)

# Build base spending array over time
annual_spending_arr = np.full(ages.shape, annual_spending)
ss_income_arr = np.zeros_like(ages, dtype=np.float64)
ss_index = np.where(ages >= social_security_start_age)
ss_income_arr[ss_index] = social_security_annual

# Add one-time events
one_time_arr = np.zeros_like(ages, dtype=np.float64)
for k, v in one_time_events.items():
    idx = np.where(ages == k)[0]
    if idx.size > 0:
        one_time_arr[idx[0]] = v

# Main simulation - build dictionary of results
results_dict = {}
for sim in range(simulations):
    wealth = starting_wealth
    sim_result = []
    for i, age in enumerate(ages):
        # Monthly contributions end at retirement
        if age < retirement_age:
            wealth += monthly_contribution * 12

        # Random investment return using normal distribution
        # This generates returns centered around avg_invest_return with standard deviation
        inv_return = np.random.normal(avg_invest_return, invest_std_dev)
        wealth *= (1 + inv_return)

        # Adjust spending for very good or bad years
        # "good" year = inv_return > avg_invest_return: increase spending; "bad" year = decrease spending
        if inv_return > avg_invest_return:
            spend = annual_spending_arr[i] * (1 + spending_volatility)
        elif inv_return < avg_invest_return:
            spend = annual_spending_arr[i] * (1 - spending_volatility)
        else:
            spend = annual_spending_arr[i]
        wealth -= spend

        # Add Social Security income if eligible
        wealth += ss_income_arr[i]

        # Charitable donation (once per year)
        wealth -= wealth * charity_pct

        # One-time events (inheritance, big purchase, etc.)
        wealth += one_time_arr[i]

        # Prevent going negative below zero
        wealth = max(wealth, 0)
        sim_result.append(wealth)

    results_dict[f"sim_{sim+1}"] = sim_result

# Create DataFrame from dictionary all at once (no fragmentation)
results = pd.DataFrame(results_dict, index=ages)
results.index.name = 'age'

# Round all values to nearest integer
results = results.round(0).astype(int)

# Save to CSV
results.to_csv("portfolio_simulation.csv")

# Calculate percentiles at each age
percentiles = pd.DataFrame(index=ages)
percentiles['p99'] = results.quantile(0.99, axis=1).round(0).astype(int)
percentiles['p95'] = results.quantile(0.95, axis=1).round(0).astype(int)
percentiles['p90'] = results.quantile(0.90, axis=1).round(0).astype(int)
percentiles['p75'] = results.quantile(0.75, axis=1).round(0).astype(int)
percentiles['median'] = results.quantile(0.50, axis=1).round(0).astype(int)
percentiles['p25'] = results.quantile(0.25, axis=1).round(0).astype(int)
percentiles['p10'] = results.quantile(0.10, axis=1).round(0).astype(int)
percentiles['p05'] = results.quantile(0.05, axis=1).round(0).astype(int)
percentiles['p01'] = results.quantile(0.01, axis=1).round(0).astype(int)

# Print summary statistics
print("\n" + "="*60)
print("MONTE CARLO SIMULATION SUMMARY")
print("="*60)
print(f"\nNumber of simulations: {simulations}")
print(f"Age range: {starting_age} to {end_age}")
print(f"Average return: {avg_invest_return*100:.1f}% per year")
print(f"Standard deviation: {invest_std_dev*100:.1f}% per year")
print("\n" + "-"*60)
print("FINAL PORTFOLIO VALUE STATISTICS (at age {})".format(end_age))
print("-"*60)
final_values = results.iloc[-1].values
print(f"99th percentile: ${int(np.percentile(final_values, 99)):>15,}")
print(f"95th percentile: ${int(np.percentile(final_values, 95)):>15,}")
print(f"90th percentile: ${int(np.percentile(final_values, 90)):>15,}")
print(f"75th percentile: ${int(np.percentile(final_values, 75)):>15,}")
print(f"Median (50th):   ${int(np.percentile(final_values, 50)):>15,}")
print(f"25th percentile: ${int(np.percentile(final_values, 25)):>15,}")
print(f"10th percentile: ${int(np.percentile(final_values, 10)):>15,}")
print(f"5th percentile:  ${int(np.percentile(final_values, 5)):>15,}")
print(f"1st percentile:  ${int(np.percentile(final_values, 1)):>15,}")
print("="*60 + "\n")

# Plot percentile bands
plt.figure(figsize=(15, 8))

# Fill between percentile bands (lightest to darkest)
plt.fill_between(ages, percentiles['p01'], percentiles['p99'], 
                 color='lightgray', alpha=0.3, label='1st-99th percentile')
plt.fill_between(ages, percentiles['p05'], percentiles['p95'], 
                 color='darkgray', alpha=0.4, label='5th-95th percentile')
plt.fill_between(ages, percentiles['p10'], percentiles['p90'], 
                 color='gray', alpha=0.5, label='10th-90th percentile')
plt.fill_between(ages, percentiles['p25'], percentiles['p75'], 
                 color='dimgray', alpha=0.6, label='25th-75th percentile')

# Plot median line
plt.plot(ages, percentiles['median'], color='black', linewidth=2.5, label='Median (50th percentile)')

# Formatting
plt.xlabel("Age", fontsize=12)
plt.ylabel("Portfolio Value ($)", fontsize=12)
plt.title("Monte Carlo Investment Portfolio Simulation - Percentile Ranges", fontsize=14, fontweight='bold')
plt.ylim(0, 10_000_000)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1_000_000:.0f}M'))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1_000_000))
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig('portfolio_simulation.png', dpi=150)
plt.show()

print("Chart saved as: portfolio_simulation.png")
