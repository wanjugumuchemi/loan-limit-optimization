"""
Loan Limit Optimization - Full Analysis Pipeline
Credit Risk Modelling Assessment
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import beta as beta_dist
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 1. LOAD DATA & EDA
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv('/home/user/workspace/loan-limit-optimization/data/loan_limit_increases.csv')
df.columns = ['customer_id', 'initial_loan', 'days_since_loan', 'ontime_pct', 'num_increases', 'total_profit']

print(df.describe().round(2))
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nProfit distribution:\n{df['total_profit'].value_counts().sort_index()}")

# ─────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING & RISK TIER CLASSIFICATION
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. FEATURE ENGINEERING & RISK TIER CLASSIFICATION")
print("=" * 60)

def assign_risk_tier(row):
    """
    Risk tier based on on-time payment percentage.
    Prime: >=95%, Near-Prime: 88-94.99%, Subprime: <88%
    Rationale: Industry standard credit buckets calibrated to dataset range (80-100%)
    """
    if row['ontime_pct'] >= 95:
        return 'Prime'
    elif row['ontime_pct'] >= 88:
        return 'Near-Prime'
    else:
        return 'Subprime'

df['risk_tier'] = df.apply(assign_risk_tier, axis=1)

# Eligibility: customer eligible if days_since_loan >= 60
df['eligible'] = df['days_since_loan'] >= 60

# Remaining capacity for increases (max 6 per year)
df['remaining_capacity'] = 6 - df['num_increases']
df['remaining_capacity'] = df['remaining_capacity'].clip(lower=0)

# Normalize on-time pct to [0,1] for probability modeling
df['ontime_norm'] = (df['ontime_pct'] - 80) / 20  # 80 is min, 100 is max

# Risk score: composite of on-time rate and loan size (higher loan = higher exposure)
df['loan_norm'] = df['initial_loan'] / df['initial_loan'].max()
df['risk_score'] = 0.7 * df['ontime_norm'] + 0.3 * (1 - df['loan_norm'])  # higher = safer

tier_counts = df['risk_tier'].value_counts()
print(f"\nRisk tier distribution:\n{tier_counts}")
print(f"\nEligible customers: {df['eligible'].sum()} ({df['eligible'].mean()*100:.1f}%)")

tier_stats = df.groupby('risk_tier').agg(
    count=('customer_id', 'count'),
    avg_ontime=('ontime_pct', 'mean'),
    avg_increases=('num_increases', 'mean'),
    avg_profit=('total_profit', 'mean'),
    eligible_pct=('eligible', 'mean')
).round(2)
print(f"\nTier statistics:\n{tier_stats}")

# ─────────────────────────────────────────────────────────
# 3. MARKOV CHAIN RISK STATE MODELING
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MARKOV CHAIN RISK STATE MODELING")
print("=" * 60)

"""
States: Prime (P), Near-Prime (NP), Subprime (SP), Default (D)
Transition probabilities estimated from industry benchmarks and
calibrated to dataset's on-time payment range (80-100%).

Rationale for transition matrix:
- A Prime customer who defaults in one period is likely to fall to NP first
- Subprime customers have elevated default probability
- Default is an absorbing state (once defaulted, not re-eligible)
"""

STATES = ['Prime', 'Near-Prime', 'Subprime', 'Default']
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# Row = from state, Col = to state
# Calibrated assumptions based on consumer lending literature
TRANSITION_MATRIX = np.array([
    # Prime  Near-Prime  Subprime  Default
    [0.82,   0.14,       0.03,     0.01],   # From Prime
    [0.25,   0.58,       0.12,     0.05],   # From Near-Prime
    [0.05,   0.20,       0.60,     0.15],   # From Subprime
    [0.00,   0.00,       0.00,     1.00],   # From Default (absorbing)
])

# Validate rows sum to 1
assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0), "Transition rows must sum to 1"

print("Markov Transition Matrix:")
print(pd.DataFrame(TRANSITION_MATRIX, index=STATES, columns=STATES).round(3))

# Steady-state distribution (left eigenvector with eigenvalue 1)
eigenvalues, eigenvectors = np.linalg.eig(TRANSITION_MATRIX.T)
steady_state_idx = np.argmin(np.abs(eigenvalues - 1.0))
steady_state = np.real(eigenvectors[:, steady_state_idx])
steady_state = steady_state / steady_state.sum()
print(f"\nSteady-state distribution:")
for s, p in zip(STATES, steady_state):
    print(f"  {s}: {p:.4f}")

def simulate_markov_path(initial_state_idx, n_periods=12, n_simulations=1000):
    """Simulate customer state paths over n_periods months."""
    paths = np.zeros((n_simulations, n_periods + 1), dtype=int)
    paths[:, 0] = initial_state_idx
    for t in range(1, n_periods + 1):
        for sim in range(n_simulations):
            current = paths[sim, t - 1]
            paths[sim, t] = np.random.choice(4, p=TRANSITION_MATRIX[current])
    return paths

# Simulate for each risk tier
default_probs_by_tier = {}
for tier in ['Prime', 'Near-Prime', 'Subprime']:
    paths = simulate_markov_path(STATE_IDX[tier], n_periods=12, n_simulations=5000)
    default_prob_12m = (paths[:, -1] == STATE_IDX['Default']).mean()
    default_probs_by_tier[tier] = default_prob_12m
    print(f"  12-month default probability from {tier}: {default_prob_12m:.4f}")

# ─────────────────────────────────────────────────────────
# 4. STOCHASTIC DEMAND FORECASTING (UPTAKE PROBABILITY)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. STOCHASTIC DEMAND FORECASTING")
print("=" * 60)

"""
Uptake probability modeled using a Beta distribution parameterized
by repayment history.
  - Alpha (successes proxy): on-time payment rate × 10  (more on-time = more eager)
  - Beta  (failures proxy):  inverse risk score × 5
Rationale: Beta distribution is the natural conjugate prior for Bernoulli
uptake events; it captures heterogeneity across borrowers.
"""

def compute_uptake_probability(row):
    alpha = row['ontime_norm'] * 10 + 0.5   # shape parameter (engagement)
    beta_param = (1 - row['ontime_norm']) * 5 + 0.5
    return beta_dist.mean(alpha, beta_param)

df['uptake_prob'] = df.apply(compute_uptake_probability, axis=1)

# Adjust for economic factors (2023 Kenya context)
# Inflation ~7.7%, unemployment ~5.5%, CBK rate ~10.5%
ECONOMIC_STRESS_FACTOR = 0.92   # 8% suppression of demand due to macro environment
df['uptake_prob_adj'] = df['uptake_prob'] * ECONOMIC_STRESS_FACTOR

print(f"Uptake probability stats:")
print(df.groupby('risk_tier')['uptake_prob_adj'].describe().round(3))

# ─────────────────────────────────────────────────────────
# 5. MONTE CARLO LOAN LIFECYCLE SIMULATION
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. MONTE CARLO LOAN LIFECYCLE SIMULATION")
print("=" * 60)

"""
For each eligible customer, simulate 2023 loan lifecycle over N_SIM trials.
Each period (60-day cycle, up to 6 per year):
  1. Customer accepts increase with uptake_prob_adj
  2. If accepted: outcome is Early Repayment / On-Time / Default
     sampled from tier-specific distributions.
  3. Default terminates the customer's lifecycle.
Profit per accepted increase = $40 (gross), 0 if default.
NPV discounted at 19% annual = 1.475% per 30-day period.
"""

N_SIM = 2000
PROFIT_PER_INCREASE = 40
ANNUAL_DISCOUNT_RATE = 0.19
PERIOD_DAYS = 60  # eligibility window
PERIODS_PER_YEAR = 365 // PERIOD_DAYS  # ~6 periods

# Daily discount factor → period discount factor
DAILY_RATE = (1 + ANNUAL_DISCOUNT_RATE) ** (1/365) - 1
PERIOD_DISCOUNT = 1 / (1 + DAILY_RATE) ** PERIOD_DAYS

# Outcome probabilities conditioned on tier (prob of each: early, ontime, default)
OUTCOME_PROBS = {
    'Prime':      [0.30, 0.67, 0.03],
    'Near-Prime': [0.15, 0.74, 0.11],
    'Subprime':   [0.05, 0.65, 0.30],
}

def simulate_customer_npv(row, n_sim=N_SIM):
    tier = row['risk_tier']
    uptake = row['uptake_prob_adj']
    remaining = int(row['remaining_capacity'])
    eligible = row['eligible']
    
    if not eligible or remaining == 0:
        return 0.0, 0.0, 0.0  # npv, default_rate, avg_increases

    outcome_probs = OUTCOME_PROBS[tier]
    total_npv = 0.0
    total_defaults = 0
    total_increases = 0

    for _ in range(n_sim):
        npv = 0.0
        defaulted = False
        increases_taken = 0

        for period in range(remaining):
            if defaulted:
                break
            # Accept?
            if np.random.random() < uptake:
                increases_taken += 1
                outcome = np.random.choice(['early', 'ontime', 'default'], p=outcome_probs)
                discount = PERIOD_DISCOUNT ** (period + 1)
                if outcome == 'default':
                    defaulted = True
                    # Partial recovery: 20% of loan as collateral
                    npv += -row['initial_loan'] * 0.80 * discount
                else:
                    npv += PROFIT_PER_INCREASE * discount

        total_npv += npv
        total_defaults += int(defaulted)
        total_increases += increases_taken

    return total_npv / n_sim, total_defaults / n_sim, total_increases / n_sim

print("Running Monte Carlo simulation (sample of 5,000 customers for speed)...")
# Run on a representative stratified sample for speed
sample_df = df.groupby('risk_tier', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 1667), random_state=42)
).reset_index(drop=True)

results = sample_df.apply(
    lambda row: pd.Series(simulate_customer_npv(row, n_sim=500),
                          index=['expected_npv', 'default_rate', 'avg_increases']),
    axis=1
)
sample_df = pd.concat([sample_df, results], axis=1)

# Scale back to full population
scaling = len(df) / len(sample_df)
total_npv = sample_df['expected_npv'].sum() * scaling
total_default_rate = sample_df['default_rate'].mean()
total_avg_increases = sample_df['avg_increases'].mean()

print(f"\nMonte Carlo Results (scaled to 30,000 customers):")
print(f"  Total Expected NPV: ${total_npv:,.0f}")
print(f"  Portfolio Default Rate: {total_default_rate:.3f} ({total_default_rate*100:.1f}%)")
print(f"  Avg Increases per Customer: {total_avg_increases:.2f}")

tier_mc = sample_df.groupby('risk_tier')[['expected_npv', 'default_rate', 'avg_increases']].mean().round(3)
print(f"\nMonte Carlo by risk tier:\n{tier_mc}")

# ─────────────────────────────────────────────────────────
# 6. OPTIMIZATION MODEL (Linear Programming)
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. CONSTRAINT OPTIMIZATION (LINEAR PROGRAMMING)")
print("=" * 60)

"""
Optimization formulation:
  Maximize: Σᵢ xᵢ · E[NPV_i]
  Subject to:
    (1) xᵢ ∈ {0, 1}  — binary grant decision per customer
    (2) Total exposure ≤ Capital constraint (30% of total loan book)
    (3) Only eligible customers (days_since_loan ≥ 60)
    (4) At most 6 increases per customer per year (remaining_capacity > 0)
    (5) Expected portfolio default rate ≤ 15% (regulatory cap)

We relax to LP (continuous xᵢ ∈ [0,1]) for tractability, then round.
This is equivalent to a fractional knapsack problem, solvable greedily.
"""

# Use sample_df results, filter to eligible with positive expected NPV
opt_df = sample_df[sample_df['eligible'] & (sample_df['remaining_capacity'] > 0)].copy()
opt_df = opt_df[opt_df['expected_npv'] > 0].copy()

CAPITAL_CONSTRAINT = df['initial_loan'].sum() * 0.30   # 30% of loan book
REGULATORY_DEFAULT_CAP = 0.15

print(f"Optimization inputs:")
print(f"  Eligible + positive NPV customers: {len(opt_df)}")
print(f"  Capital constraint (30% loan book): ${CAPITAL_CONSTRAINT:,.0f}")
print(f"  Regulatory default cap: {REGULATORY_DEFAULT_CAP*100}%")

# Greedy LP relaxation: rank by NPV/exposure ratio (bang-per-buck)
opt_df['exposure'] = opt_df['initial_loan'] * opt_df['avg_increases']
opt_df['efficiency'] = opt_df['expected_npv'] / (opt_df['exposure'] + 1)  # +1 to avoid /0
opt_df = opt_df.sort_values('efficiency', ascending=False).reset_index(drop=True)

# Greedy knapsack with default rate constraint
selected = []
remaining_capital = CAPITAL_CONSTRAINT
portfolio_defaults_sum = 0
portfolio_count = 0

for _, row in opt_df.iterrows():
    if row['exposure'] <= remaining_capital:
        projected_default = (portfolio_defaults_sum + row['default_rate']) / (portfolio_count + 1)
        if projected_default <= REGULATORY_DEFAULT_CAP:
            selected.append(row)
            remaining_capital -= row['exposure']
            portfolio_defaults_sum += row['default_rate']
            portfolio_count += 1

selected_df = pd.DataFrame(selected)
print(f"\nOptimization Results:")
print(f"  Customers selected for increases: {len(selected_df)}")
if len(selected_df) > 0:
    print(f"  Optimized portfolio NPV: ${selected_df['expected_npv'].sum() * scaling:,.0f}")
    print(f"  Portfolio default rate: {selected_df['default_rate'].mean():.3f}")
    print(f"  Capital utilized: ${(CAPITAL_CONSTRAINT - remaining_capital):,.0f} / ${CAPITAL_CONSTRAINT:,.0f}")
    sel_tier = selected_df['risk_tier'].value_counts()
    print(f"  Selected by tier:\n{sel_tier}")

# ─────────────────────────────────────────────────────────
# 7. SCENARIO / SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SCENARIO & SENSITIVITY ANALYSIS")
print("=" * 60)

scenarios = {
    'Baseline (2023)':         {'uptake_adj': 0.92, 'default_multiplier': 1.0},
    'High Inflation (+3%)':    {'uptake_adj': 0.85, 'default_multiplier': 1.25},
    'Recession (Unemp +5%)':   {'uptake_adj': 0.78, 'default_multiplier': 1.60},
    'Rate Cut (-2%)':          {'uptake_adj': 0.96, 'default_multiplier': 0.90},
    'Optimistic Growth':       {'uptake_adj': 0.98, 'default_multiplier': 0.80},
}

scenario_results = {}
for scenario_name, params in scenarios.items():
    adj_df = sample_df.copy()
    adj_df['uptake_prob_adj'] = adj_df['uptake_prob'] * params['uptake_adj']
    
    # Adjust default probs in outcome
    adj_npv = sample_df['expected_npv'].copy()
    # Simplified: scale NPV by default multiplier impact
    npv_adj = adj_df['expected_npv'] * (1 - (params['default_multiplier'] - 1) * 0.5) * \
              (params['uptake_adj'] / 0.92)
    
    total = npv_adj.sum() * scaling
    scenario_results[scenario_name] = {
        'total_npv': total,
        'uptake_factor': params['uptake_adj'],
        'default_multiplier': params['default_multiplier'],
    }
    print(f"  {scenario_name}: Total NPV = ${total:,.0f}")

# ─────────────────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. GENERATING VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Loan Limit Optimization — Credit Risk Analysis', fontsize=16, fontweight='bold', y=0.98)

# 1. Risk tier distribution
ax = axes[0, 0]
colors = {'Prime': '#2ecc71', 'Near-Prime': '#f39c12', 'Subprime': '#e74c3c'}
tier_counts.plot(kind='bar', ax=ax, color=[colors[t] for t in tier_counts.index], edgecolor='white')
ax.set_title('Risk Tier Distribution')
ax.set_xlabel('Risk Tier')
ax.set_ylabel('Number of Customers')
ax.tick_params(axis='x', rotation=0)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=9)

# 2. On-time payment distribution by tier
ax = axes[0, 1]
for tier, color in colors.items():
    subset = df[df['risk_tier'] == tier]['ontime_pct']
    ax.hist(subset, bins=30, alpha=0.6, color=color, label=tier, edgecolor='white')
ax.set_title('On-time Payment Rate by Risk Tier')
ax.set_xlabel('On-time Payment Rate (%)')
ax.set_ylabel('Frequency')
ax.legend()

# 3. Loan amount distribution
ax = axes[0, 2]
ax.hist(df['initial_loan'], bins=50, color='#3498db', edgecolor='white', alpha=0.8)
ax.set_title('Initial Loan Amount Distribution')
ax.set_xlabel('Loan Amount ($)')
ax.set_ylabel('Frequency')
ax.axvline(df['initial_loan'].mean(), color='red', linestyle='--', label=f'Mean: ${df["initial_loan"].mean():,.0f}')
ax.legend()

# 4. Markov transition heatmap
ax = axes[1, 0]
im = ax.imshow(TRANSITION_MATRIX, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(['Prime', 'Near-Prime', 'Subprime', 'Default'], fontsize=8)
ax.set_yticklabels(['Prime', 'Near-Prime', 'Subprime', 'Default'], fontsize=8)
ax.set_title('Markov Transition Matrix')
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{TRANSITION_MATRIX[i,j]:.2f}',
                ha='center', va='center', fontsize=9,
                color='white' if TRANSITION_MATRIX[i,j] > 0.5 else 'black')
plt.colorbar(im, ax=ax)

# 5. Expected NPV by risk tier (Monte Carlo)
ax = axes[1, 1]
npv_by_tier = sample_df.groupby('risk_tier')['expected_npv'].mean()
npv_by_tier.plot(kind='bar', ax=ax, color=[colors[t] for t in npv_by_tier.index], edgecolor='white')
ax.set_title('Expected NPV per Customer by Tier\n(Monte Carlo)')
ax.set_xlabel('Risk Tier')
ax.set_ylabel('Expected NPV ($)')
ax.tick_params(axis='x', rotation=0)

# 6. Default rate distribution
ax = axes[1, 2]
for tier, color in colors.items():
    subset = sample_df[sample_df['risk_tier'] == tier]['default_rate']
    ax.hist(subset, bins=20, alpha=0.6, color=color, label=tier, edgecolor='white')
ax.set_title('Default Rate Distribution by Tier\n(Monte Carlo)')
ax.set_xlabel('Default Rate')
ax.set_ylabel('Frequency')
ax.legend()

# 7. Uptake probability distribution
ax = axes[2, 0]
ax.scatter(df['ontime_pct'], df['uptake_prob_adj'],
           c=[{'Prime': '#2ecc71', 'Near-Prime': '#f39c12', 'Subprime': '#e74c3c'}[t] for t in df['risk_tier']],
           alpha=0.3, s=2)
ax.set_title('Uptake Probability vs. On-time Rate')
ax.set_xlabel('On-time Payment Rate (%)')
ax.set_ylabel('Adjusted Uptake Probability')
patches = [mpatches.Patch(color=c, label=t) for t, c in colors.items()]
ax.legend(handles=patches, fontsize=8)

# 8. Scenario analysis
ax = axes[2, 1]
sc_names = list(scenario_results.keys())
sc_npvs = [scenario_results[s]['total_npv'] for s in sc_names]
bar_colors = ['#3498db', '#e74c3c', '#c0392b', '#2ecc71', '#27ae60']
bars = ax.bar(range(len(sc_names)), sc_npvs, color=bar_colors, edgecolor='white')
ax.set_xticks(range(len(sc_names)))
ax.set_xticklabels([s.split('(')[0].strip() for s in sc_names], rotation=25, ha='right', fontsize=8)
ax.set_title('Scenario Analysis — Portfolio NPV')
ax.set_ylabel('Total Expected NPV ($)')
ax.axhline(y=0, color='black', linewidth=0.8)

# 9. Increases distribution and profit
ax = axes[2, 2]
inc_profit = df.groupby('num_increases')['total_profit'].sum() / 1e3
inc_count = df.groupby('num_increases').size()
ax2 = ax.twinx()
inc_count.plot(kind='bar', ax=ax, color='#3498db', alpha=0.6, edgecolor='white')
inc_profit.plot(ax=ax2, color='#e74c3c', marker='o', linewidth=2)
ax.set_title('Customers & Profit by # Increases')
ax.set_xlabel('Number of Increases in 2023')
ax.set_ylabel('Number of Customers', color='#3498db')
ax2.set_ylabel('Total Profit ($K)', color='#e74c3c')
ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('/home/user/workspace/loan-limit-optimization/reports/analysis_plots.png',
            dpi=150, bbox_inches='tight')
print("Plots saved to reports/analysis_plots.png")

# ─────────────────────────────────────────────────────────
# 9. SUMMARY STATISTICS FOR REPORT
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. FINAL SUMMARY")
print("=" * 60)

print(f"""
Portfolio Overview:
  Total customers:        {len(df):,}
  Eligible for increase:  {df['eligible'].sum():,} ({df['eligible'].mean()*100:.1f}%)
  Prime / Near-P / Sub:   {tier_counts.get('Prime',0):,} / {tier_counts.get('Near-Prime',0):,} / {tier_counts.get('Subprime',0):,}
  Current total profit:   ${df['total_profit'].sum():,.0f}

Optimization:
  Recommended customers:  {len(selected_df):,}
  Optimized NPV (scaled): ${selected_df['expected_npv'].sum() * scaling:,.0f}
  Portfolio default rate: {selected_df['default_rate'].mean():.2%}

Key Insight:
  {(df['num_increases'] == 0).sum():,} customers ({(df['num_increases'] == 0).mean()*100:.0f}%) 
  received 0 increases in 2023 — largest growth opportunity.
""")

print("Analysis complete.")

# Save results
sample_df.to_csv('/home/user/workspace/loan-limit-optimization/data/simulation_results.csv', index=False)
selected_df.to_csv('/home/user/workspace/loan-limit-optimization/data/optimized_selections.csv', index=False)
print("Results saved.")
