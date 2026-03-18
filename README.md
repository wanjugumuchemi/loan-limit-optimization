# Loan Limit Optimization — Credit Risk Modelling Assessment

> Algorithmic approach to optimize loan limit increases for a 30,000-customer portfolio in 2023, maximizing expected profitability while controlling default risk under regulatory and capital constraints.

## Overview

This project addresses a core challenge in consumer lending: **when and to whom should a lender offer loan limit increases?** A naive "offer everyone" strategy ignores default risk and destroys value. A "offer no one" strategy leaves significant revenue unrealized. The optimal policy lies between these extremes — and requires probabilistic modeling to find.

## Tech Stack

- **Python 3.13** | pandas | numpy | scipy | matplotlib | seaborn
- **Methods:** Markov Chains, Monte Carlo Simulation, Beta Distribution Demand Modeling, Greedy LP Knapsack Optimization
- **IDE:** VS Code

## Repository Structure

```
loan-limit-optimization/
├── data/
│   ├── loan_limit_increases.csv      # Source dataset (30,000 records)
│   ├── simulation_results.csv        # Monte Carlo output per customer
│   └── optimized_selections.csv      # Final optimization selection
├── notebooks/
│   └── loan_limit_optimization.ipynb # Full analysis notebook
├── reports/
│   ├── analysis_dashboard.png        # 9-panel visualization dashboard
│   ├── report.md                     # Comprehensive methodology report
│   └── metrics.json                  # Key output metrics
├── src/
│   └── full_analysis.py              # Standalone Python script
├── requirements.txt
└── README.md
```

## Methodology Summary

| Step | Method | Purpose |
|------|--------|---------|
| 1 | EDA | Understand data distributions, identify 44% zero-increase opportunity |
| 2 | Risk Tier Classification | Segment customers: Prime ≥95%, Near-Prime 88–94.99%, Subprime <88% |
| 3 | Markov Chain Modeling | Model customer transitions between risk states over time |
| 4 | Beta Distribution Demand Forecasting | Stochastic uptake probability per customer |
| 5 | Monte Carlo Simulation | 600-trial lifecycle simulation; expected NPV & default rate per customer |
| 6 | Greedy LP Optimization | Maximize portfolio NPV subject to capital + regulatory constraints |
| 7 | Scenario Analysis | Stress-test under inflation, recession, rate cuts, growth scenarios |

## Key Results

| Metric | Value |
|--------|-------|
| Recommended customers (scaled) | ~2,243 of 25,068 eligible |
| Optimized portfolio NPV | $84,006 (baseline 2023) |
| Portfolio default rate | 8.2% (vs. 15% regulatory cap) |
| Capital utilization | 4.6% of $24.77M constraint |
| Worst-case NPV (recession) | $49,866 (still positive) |

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/wanjugumuchemi/loan-limit-optimization.git
cd loan-limit-optimization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full analysis
python src/full_analysis.py

# 4. Launch Jupyter notebook
jupyter notebook notebooks/loan_limit_optimization.ipynb
```

## Key Assumptions

- Loss Given Default (LGD): 80% (20% collateral recovery)
- Macro demand suppression factor: 0.92 (Kenya 2023 economic conditions)
- Regulatory default cap: 15% of selected portfolio
- Capital constraint: 30% of total loan book ($82.6M → $24.8M cap)
- Eligibility: 60-day seasoning post-disbursement + on-time payments

## License

MIT License — Academic/Assessment use.
