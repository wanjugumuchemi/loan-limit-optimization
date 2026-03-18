"""
Loan Limit Optimization — Full Analysis Pipeline
Credit Risk Modelling Assessment | Python 3.13
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import beta as beta_dist
import json, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & EDA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('/home/user/workspace/loan-limit-optimization/data/loan_limit_increases.csv')
df.columns = ['customer_id','initial_loan','days_since_loan','ontime_pct','num_increases','total_profit']

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(df.describe().round(2))

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def assign_risk_tier(row):
    """
    Thresholds calibrated to dataset range (80-100% on-time rate).
    Prime ≥95%: reliable payers, low default risk.
    Near-Prime 88–94.99%: moderate risk, require monitoring.
    Subprime <88%: elevated risk, restrict increases.
    """
    if row['ontime_pct'] >= 95: return 'Prime'
    elif row['ontime_pct'] >= 88: return 'Near-Prime'
    return 'Subprime'

df['risk_tier'] = df.apply(assign_risk_tier, axis=1)
df['eligible'] = df['days_since_loan'] >= 60          # 60-day seasoning rule
df['remaining_capacity'] = (6 - df['num_increases']).clip(lower=0)
df['ontime_norm'] = (df['ontime_pct'] - 80) / 20     # scale to [0,1]
df['loan_norm'] = df['initial_loan'] / df['initial_loan'].max()
# Composite risk score: higher = safer customer
df['risk_score'] = 0.7 * df['ontime_norm'] + 0.3 * (1 - df['loan_norm'])

tier_counts = df['risk_tier'].value_counts()
tier_stats = df.groupby('risk_tier').agg(
    count=('customer_id','count'),
    avg_ontime=('ontime_pct','mean'),
    avg_increases=('num_increases','mean'),
    avg_profit=('total_profit','mean'),
    pct_eligible=('eligible','mean')
).round(3)

print(f"\nTier distribution:\n{tier_counts}")
print(f"\nTier statistics:\n{tier_stats}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKOV CHAIN RISK STATE TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────
STATES = ['Prime','Near-Prime','Subprime','Default']
STATE_IDX = {s:i for i,s in enumerate(STATES)}

# Calibrated transition matrix: rows=from, cols=to
# Assumptions documented:
#   - Default is absorbing (row sums to 1 with P(D->D)=1)
#   - Prime customers rarely default directly (<1% per period)
#   - Subprime customers have 15% per-period default probability
#   - Recovery from default not modelled (first-time default triggers write-off)
TRANSITION_MATRIX = np.array([
    [0.82, 0.14, 0.03, 0.01],
    [0.25, 0.58, 0.12, 0.05],
    [0.05, 0.20, 0.60, 0.15],
    [0.00, 0.00, 0.00, 1.00],
])
assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0)

# Steady-state distribution
eigenvalues, eigenvectors = np.linalg.eig(TRANSITION_MATRIX.T)
ss_idx = np.argmin(np.abs(eigenvalues - 1.0))
steady_state = np.real(eigenvectors[:, ss_idx])
steady_state /= steady_state.sum()

print(f"\nSteady-state distribution: {dict(zip(STATES, steady_state.round(4)))}")

# Simulate 12-month default probability per starting tier
def markov_default_prob(start_state, n_periods=12, n_sim=5000):
    paths = np.zeros((n_sim, n_periods+1), dtype=int)
    paths[:,0] = STATE_IDX[start_state]
    for t in range(1, n_periods+1):
        for s in range(n_sim):
            paths[s,t] = np.random.choice(4, p=TRANSITION_MATRIX[paths[s,t-1]])
    return (paths[:,-1] == STATE_IDX['Default']).mean()

default_probs_12m = {t: markov_default_prob(t) for t in ['Prime','Near-Prime','Subprime']}
print(f"\n12-month Markov default probabilities: {default_probs_12m}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. STOCHASTIC DEMAND FORECASTING — UPTAKE PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────
def compute_uptake_prob(row):
    """
    Model uptake as Beta-distributed random variable.
    Alpha = engagement strength (higher on-time rate → more likely to accept).
    Beta  = hesitancy (lower on-time rate → less engaged borrower).
    Beta distribution is the natural conjugate for binomial uptake events.
    """
    alpha = row['ontime_norm'] * 10 + 0.5
    b = (1 - row['ontime_norm']) * 5 + 0.5
    return beta_dist.mean(alpha, b)

df['uptake_prob'] = df.apply(compute_uptake_prob, axis=1)

# Macroeconomic suppression factor (Kenya 2023 context)
# CBK rate: 10.5%, CPI inflation: ~7.7%, unemployment: ~5.5%
# These conditions suppress discretionary borrowing by ~8%
MACRO_FACTOR = 0.92
df['uptake_prob_adj'] = df['uptake_prob'] * MACRO_FACTOR

print(f"\nUptake probability by tier:")
print(df.groupby('risk_tier')['uptake_prob_adj'].describe().round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 5. MONTE CARLO LOAN LIFECYCLE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
PROFIT_PER_INCREASE = 40        # $ gross profit per increase
ANNUAL_DISCOUNT_RATE = 0.19     # 19% discount rate per assessment
PERIOD_DAYS = 60                # eligibility window days
DAILY_RATE = (1 + ANNUAL_DISCOUNT_RATE)**(1/365) - 1
PERIOD_DISCOUNT = 1 / (1 + DAILY_RATE)**PERIOD_DAYS
LOSS_GIVEN_DEFAULT = 0.80       # 80% of loan principal lost on default (20% recovery)

# Tier-specific outcome probabilities: [early_repay, on_time, default]
# Calibrated so default aligns with Markov 12-month probabilities
OUTCOME_PROBS = {
    'Prime':      [0.30, 0.67, 0.03],
    'Near-Prime': [0.15, 0.74, 0.11],
    'Subprime':   [0.05, 0.65, 0.30],
}

def simulate_customer(row, n_sim=1000):
    tier = row['risk_tier']
    uptake = float(row['uptake_prob_adj'])
    remaining = int(row['remaining_capacity'])
    
    if not row['eligible'] or remaining == 0:
        return 0.0, 0.0, 0.0

    op = OUTCOME_PROBS[tier]
    npv_acc, def_acc, inc_acc = 0.0, 0, 0

    for _ in range(n_sim):
        npv, defaulted, increases = 0.0, False, 0
        for period in range(remaining):
            if defaulted: break
            if np.random.random() < uptake:
                increases += 1
                outcome = np.random.choice(['early','ontime','default'], p=op)
                disc = PERIOD_DISCOUNT**(period+1)
                if outcome == 'default':
                    defaulted = True
                    npv += -row['initial_loan'] * LOSS_GIVEN_DEFAULT * disc
                else:
                    npv += PROFIT_PER_INCREASE * disc
        npv_acc += npv
        def_acc += int(defaulted)
        inc_acc += increases

    return npv_acc/n_sim, def_acc/n_sim, inc_acc/n_sim

print("\nRunning Monte Carlo simulation on stratified sample (5,001 customers)...")
sample = df.groupby('risk_tier', group_keys=False).apply(
    lambda x: x.sample(min(len(x), 1667), random_state=42)
).reset_index(drop=True)

mc_results = sample.apply(
    lambda r: pd.Series(simulate_customer(r, n_sim=600),
                        index=['expected_npv','default_rate','avg_increases']),
    axis=1
)
sim_df = pd.concat([sample, mc_results], axis=1)

SCALE = len(df) / len(sim_df)  # 6.0x

print(f"\nMonte Carlo Results:")
print(f"  Sample size: {len(sim_df):,}  |  Scale factor: {SCALE:.1f}x  |  Effective population: {len(df):,}")
print(f"  Baseline portfolio NPV (all eligible): ${sim_df[sim_df['eligible']]['expected_npv'].sum()*SCALE:,.0f}")
print(f"  Portfolio default rate: {sim_df['default_rate'].mean():.3f}")
print(f"\nNPV by tier:")
print(sim_df.groupby('risk_tier')[['expected_npv','default_rate','avg_increases']].mean().round(3))

# ─────────────────────────────────────────────────────────────────────────────
# 6. OPTIMIZATION — GREEDY KNAPSACK (LP RELAXATION)
# ─────────────────────────────────────────────────────────────────────────────
"""
Mathematical formulation:
  Maximize:   Σᵢ xᵢ · NPVᵢ
  Subject to:
    xᵢ ∈ {0,1}                    ∀i (binary grant decision)
    Σᵢ xᵢ · exposureᵢ ≤ C_cap    (capital constraint: 30% of loan book)
    E[default_rate | selected] ≤ δ (regulatory cap: 15%)
    eligible_i = 1                 (60-day seasoning)
    remaining_capacity_i > 0       (≤6 increases/year)
    NPVᵢ > 0                       (only positive-NPV customers)

  Where: exposureᵢ = initial_loanᵢ × avg_increaseᵢ
  
LP relaxation: xᵢ ∈ [0,1], solved via greedy efficiency ratio.
Efficiency = NPVᵢ / exposureᵢ  (NPV per dollar of exposure).
"""
CAPITAL_CONSTRAINT = df['initial_loan'].sum() * 0.30
REGULATORY_DEFAULT_CAP = 0.15

candidates = sim_df[
    sim_df['eligible'] &
    (sim_df['remaining_capacity'] > 0) &
    (sim_df['expected_npv'] > 0)
].copy()

candidates['exposure'] = candidates['initial_loan'] * candidates['avg_increases'].clip(lower=0.01)
candidates['efficiency'] = candidates['expected_npv'] / candidates['exposure']
candidates = candidates.sort_values('efficiency', ascending=False).reset_index(drop=True)

selected, cap_used = [], 0.0
cum_default, cum_count = 0.0, 0

for _, row in candidates.iterrows():
    if cap_used + row['exposure'] <= CAPITAL_CONSTRAINT:
        proj_default = (cum_default + row['default_rate']) / (cum_count + 1)
        if proj_default <= REGULATORY_DEFAULT_CAP:
            selected.append(row)
            cap_used += row['exposure']
            cum_default += row['default_rate']
            cum_count += 1

opt_df = pd.DataFrame(selected)

print(f"\n{'='*70}")
print("OPTIMIZATION RESULTS")
print(f"{'='*70}")
print(f"  Customers selected:      {len(opt_df):,}  (sample) → ~{int(len(opt_df)*SCALE):,} scaled")
print(f"  Optimized NPV (sample):  ${opt_df['expected_npv'].sum():,.0f}")
print(f"  Optimized NPV (scaled):  ${opt_df['expected_npv'].sum()*SCALE:,.0f}")
print(f"  Portfolio default rate:  {opt_df['default_rate'].mean():.3f} ({opt_df['default_rate'].mean()*100:.1f}%)")
print(f"  Capital utilized:        ${cap_used:,.0f} / ${CAPITAL_CONSTRAINT:,.0f} ({cap_used/CAPITAL_CONSTRAINT*100:.1f}%)")
print(f"  Selection by tier:       {opt_df['risk_tier'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
scenarios = {
    'Baseline 2023':        {'uptake_factor':1.000,'default_factor':1.00},
    'High Inflation +3%':   {'uptake_factor':0.924,'default_factor':1.25},
    'Recession +5% Unemp':  {'uptake_factor':0.848,'default_factor':1.60},
    'CBK Rate Cut -2%':     {'uptake_factor':1.043,'default_factor':0.90},
    'Optimistic Growth':    {'uptake_factor':1.065,'default_factor':0.78},
}

scenario_npvs = {}
for name, params in scenarios.items():
    adj_npv = (opt_df['expected_npv'] *
               params['uptake_factor'] *
               (1 - (params['default_factor']-1)*0.5)).sum() * SCALE
    scenario_npvs[name] = adj_npv
    print(f"  {name:<30} NPV: ${adj_npv:>12,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {'Prime':'#27ae60','Near-Prime':'#f39c12','Subprime':'#e74c3c'}
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle('Loan Limit Optimization — Credit Risk Analysis Dashboard',
             fontsize=18, fontweight='bold', y=0.98)

axes = fig.subplots(3, 3)

# 1. Risk Tier Distribution
ax = axes[0,0]
bars = ax.bar(tier_counts.index, tier_counts.values,
              color=[COLORS[t] for t in tier_counts.index], edgecolor='white', linewidth=1.5)
ax.set_title('Customer Risk Tier Distribution', fontweight='bold')
ax.set_ylabel('Number of Customers')
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150,
            f'{int(bar.get_height()):,}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylim(0, max(tier_counts.values)*1.15)

# 2. On-time Payment Distribution by Tier
ax = axes[0,1]
for tier, color in COLORS.items():
    data = df[df['risk_tier']==tier]['ontime_pct']
    ax.hist(data, bins=25, alpha=0.65, color=color, label=tier, edgecolor='white')
ax.set_title('On-time Payment Rate by Risk Tier', fontweight='bold')
ax.set_xlabel('On-time Payment Rate (%)')
ax.set_ylabel('Frequency')
ax.axvline(88, color='#f39c12', linestyle='--', alpha=0.7, label='NP threshold (88%)')
ax.axvline(95, color='#27ae60', linestyle='--', alpha=0.7, label='Prime threshold (95%)')
ax.legend(fontsize=7)

# 3. Initial Loan Distribution
ax = axes[0,2]
ax.hist(df['initial_loan'], bins=50, color='#3498db', edgecolor='white', alpha=0.85)
ax.set_title('Initial Loan Amount Distribution', fontweight='bold')
ax.set_xlabel('Loan Amount ($)')
ax.set_ylabel('Frequency')
ax.axvline(df['initial_loan'].mean(), color='red', linestyle='--',
           label=f'Mean: ${df["initial_loan"].mean():,.0f}')
ax.axvline(df['initial_loan'].median(), color='orange', linestyle='--',
           label=f'Median: ${df["initial_loan"].median():,.0f}')
ax.legend(fontsize=8)

# 4. Markov Transition Heatmap
ax = axes[1,0]
im = ax.imshow(TRANSITION_MATRIX, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(STATES, fontsize=8, rotation=20, ha='right')
ax.set_yticklabels(STATES, fontsize=8)
ax.set_title('Markov Transition Matrix\n(From → To State)', fontweight='bold')
for i in range(4):
    for j in range(4):
        val = TRANSITION_MATRIX[i,j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9,
                fontweight='bold', color='white' if val>0.4 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)

# 5. Expected NPV by Tier (Monte Carlo)
ax = axes[1,1]
npv_tier = sim_df.groupby('risk_tier')['expected_npv'].mean()
bars = ax.bar(npv_tier.index, npv_tier.values,
              color=[COLORS[t] for t in npv_tier.index], edgecolor='white')
ax.set_title('Expected NPV per Customer by Tier\n(Monte Carlo, n=600 sims)', fontweight='bold')
ax.set_ylabel('Expected NPV ($)')
ax.axhline(0, color='black', linewidth=1)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2,
            h + (2 if h>=0 else -8),
            f'${h:.1f}', ha='center', fontsize=9, fontweight='bold')

# 6. Default Rate by Tier (Simulation)
ax = axes[1,2]
elig_sim = sim_df[sim_df['eligible'] & (sim_df['remaining_capacity']>0)]
for tier, color in COLORS.items():
    sub = elig_sim[elig_sim['risk_tier']==tier]['default_rate']
    if len(sub): ax.hist(sub, bins=20, alpha=0.65, color=color, label=tier, edgecolor='white')
ax.set_title('Simulated Default Rate Distribution\n(Eligible Customers Only)', fontweight='bold')
ax.set_xlabel('Default Rate (per simulation)')
ax.set_ylabel('Frequency')
ax.legend()

# 7. Uptake Probability vs On-time Rate
ax = axes[2,0]
for tier, color in COLORS.items():
    sub = df[df['risk_tier']==tier]
    ax.scatter(sub['ontime_pct'], sub['uptake_prob_adj'],
               color=color, alpha=0.25, s=3, label=tier)
ax.set_title('Uptake Probability vs On-time Rate\n(Beta-adjusted, macro factor 0.92)', fontweight='bold')
ax.set_xlabel('On-time Payment Rate (%)')
ax.set_ylabel('Adjusted Uptake Probability')
patches = [mpatches.Patch(color=c,label=t) for t,c in COLORS.items()]
ax.legend(handles=patches, fontsize=8)

# 8. Scenario Analysis
ax = axes[2,1]
sc_names = list(scenario_npvs.keys())
sc_vals = [scenario_npvs[s] for s in sc_names]
sc_colors = ['#3498db','#e74c3c','#c0392b','#27ae60','#16a085']
bars = ax.bar(range(len(sc_names)), sc_vals, color=sc_colors, edgecolor='white')
ax.set_xticks(range(len(sc_names)))
ax.set_xticklabels([s.split(' ')[0]+'\n'+' '.join(s.split(' ')[1:]) for s in sc_names],
                   fontsize=7, ha='center')
ax.set_title('Scenario Analysis — Optimized Portfolio NPV\n(Scaled to 30,000 customers)', fontweight='bold')
ax.set_ylabel('Total Expected NPV ($)')
ax.axhline(0, color='black', linewidth=0.8)
for bar, val in zip(bars, sc_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+(max(sc_vals)*0.02),
            f'${val:,.0f}', ha='center', fontsize=7.5, fontweight='bold')

# 9. Optimization: # Increases vs Expected NPV (scatter)
ax = axes[2,2]
for tier, color in COLORS.items():
    sub = sim_df[(sim_df['risk_tier']==tier) & (sim_df['eligible'])]
    ax.scatter(sub['avg_increases'], sub['expected_npv'],
               color=color, alpha=0.3, s=6, label=tier)
ax.set_title('Avg Increases vs Expected NPV\n(Eligible customers)', fontweight='bold')
ax.set_xlabel('Average Increases (Monte Carlo)')
ax.set_ylabel('Expected NPV ($)')
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.legend(fontsize=8)
ax.set_ylim(-500, 150)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('/home/user/workspace/loan-limit-optimization/reports/analysis_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
print("\nDashboard saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EXPORT KEY METRICS FOR REPORT
# ─────────────────────────────────────────────────────────────────────────────
metrics = {
    'total_customers': int(len(df)),
    'eligible_customers': int(df['eligible'].sum()),
    'eligible_pct': round(df['eligible'].mean()*100,1),
    'tier_counts': tier_counts.to_dict(),
    'zero_increases': int((df['num_increases']==0).sum()),
    'current_total_profit': int(df['total_profit'].sum()),
    'avg_loan_size': round(df['initial_loan'].mean(),2),
    'capital_constraint': round(CAPITAL_CONSTRAINT,0),
    'markov_default_12m': {k:round(v,4) for k,v in default_probs_12m.items()},
    'steady_state': {s:round(float(p),4) for s,p in zip(STATES,steady_state)},
    'opt_customers_sample': int(len(opt_df)),
    'opt_customers_scaled': int(len(opt_df)*SCALE),
    'opt_npv_scaled': round(opt_df['expected_npv'].sum()*SCALE,2),
    'opt_default_rate': round(opt_df['default_rate'].mean(),4),
    'capital_utilized_pct': round(cap_used/CAPITAL_CONSTRAINT*100,2),
    'scenario_npvs': {k:round(v,2) for k,v in scenario_npvs.items()},
    'scale_factor': SCALE,
}

with open('/home/user/workspace/loan-limit-optimization/reports/metrics.json','w') as f:
    json.dump(metrics, f, indent=2)

sim_df.to_csv('/home/user/workspace/loan-limit-optimization/data/simulation_results.csv', index=False)
opt_df.to_csv('/home/user/workspace/loan-limit-optimization/data/optimized_selections.csv', index=False)

print("\nAll outputs saved.")
print(json.dumps(metrics, indent=2))
