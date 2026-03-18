# Loan Limit Optimization: Methodology, Assumptions & Insights

**Author:** raremuchie@gmail.com  
**Date:** March 2026  
**Dataset:** 30,000 customer loan records (2023)

---

## 1. Problem Framing

The core decision problem is a **stochastic dynamic optimization**: for each customer at each eligibility window, decide whether to offer a loan limit increase. The decision must account for:

- **Heterogeneous risk**: customers differ in repayment reliability
- **Stochastic outcomes**: even identical customers may repay or default with some probability
- **Temporal constraints**: 60-day seasoning, ≤6 increases/year, capital limits
- **Asymmetric payoffs**: +$40 profit per successful increase vs. −$2,200 average loss on default ($2,752 × 80% LGD)

This asymmetry (50:1 loss-to-gain ratio at average loan size) means default probability must be kept extremely low for the $40/increase model to generate positive expected value.

---

## 2. Dataset Analysis

### 2.1 Features

| Column | Range | Mean | Notes |
|--------|-------|------|-------|
| Initial Loan ($) | 500–4,999 | 2,753 | Uniform-ish, slight right skew |
| Days Since Last Loan | 0–364 | 181 | Most customers well past 60-day threshold |
| On-time Payment (%) | 80–100 | 90.0 | Dataset excludes complete defaulters (all ≥80%) |
| # Increases 2023 | 0–5 | 2.24 | Historical max = 5, policy max = 6 |
| Total Profit ($) | 0–120 | 44.72 | $40 × num_increases |

### 2.2 Critical Finding: Zero-Increase Segment

**13,207 customers (44%)** had zero increases in 2023. This is the largest single revenue opportunity. However, the Monte Carlo simulation reveals that offering increases to all zero-increase customers without risk screening would destroy portfolio value — most zero-increase customers are Subprime with negative expected NPV.

---

## 3. Risk Tier Classification

### 3.1 Threshold Rationale

Thresholds were calibrated to the dataset's on-time payment range (80–100%):

| Tier | Threshold | Population | Avg On-time | 12-month Default (Markov) |
|------|-----------|-----------|-------------|--------------------------|
| Prime | ≥95% | 7,520 (25.1%) | 97.5% | 33.1% |
| Near-Prime | 88–94.99% | 10,622 (35.4%) | 91.5% | 41.6% |
| Subprime | <88% | 11,858 (39.5%) | 84.0% | 59.7% |

Note: The 12-month Markov default probabilities appear high because the Markov chain allows cascading transitions through intermediate states. The **per-period** (60-day) default probability is much lower (3%, 11%, 30% for Prime/Near-Prime/Subprime respectively).

---

## 4. Markov Chain Model

### 4.1 States
- **Prime**: On-time payment rate ≥95%
- **Near-Prime**: On-time payment rate 88–94.99%  
- **Subprime**: On-time payment rate <88%
- **Default**: Terminal absorbing state

### 4.2 Transition Matrix

```
From\To    Prime  Near-Prime  Subprime  Default
Prime      0.82   0.14        0.03      0.01
Near-Prime 0.25   0.58        0.12      0.05
Subprime   0.05   0.20        0.60      0.15
Default    0.00   0.00        0.00      1.00
```

### 4.3 Key Properties
- Default is **absorbing** — consistent with first-time default write-off treatment
- Prime customers have strong self-reinforcing behavior (82% stay Prime)
- Subprime customers have significant mobility: 25% recover to Near-Prime or Prime in 3 months
- Long-run steady state: all mass at Default (inevitable without credit policy intervention)

---

## 5. Demand Forecasting

### 5.1 Beta Distribution Parameterization

$$P(\text{accept}_i) \sim \text{Beta}(\alpha_i, \beta_i)$$

$$\alpha_i = \text{ontime\_norm}_i \times 10 + 0.5$$
$$\beta_i = (1 - \text{ontime\_norm}_i) \times 5 + 0.5$$

**Why Beta?** The Beta distribution is:
1. Bounded on [0,1] — appropriate for probability modeling
2. Flexible: can be symmetric, left-skewed, or right-skewed depending on parameters
3. The natural conjugate prior for Bernoulli uptake events — allows Bayesian updating

### 5.2 Macroeconomic Adjustment

Kenya 2023 economic context suppresses borrowing demand by an estimated 8%:
- **CBK rate**: 10.5% (elevated credit cost)
- **CPI inflation**: 7.7% (erodes real income)
- **Unemployment**: 5.5% (moderate but rising)

Applied **macro factor = 0.92** to all uptake probabilities.

---

## 6. Monte Carlo Simulation

### 6.1 Per-Period Outcome Probabilities

| Tier | Early Repay | On-Time | Default |
|------|:-----------:|:-------:|:-------:|
| Prime | 30% | 67% | 3% |
| Near-Prime | 15% | 74% | 11% |
| Subprime | 5% | 65% | 30% |

### 6.2 NPV Calculation

Each $40 profit is discounted to present value:

$$\text{NPV}_t = \frac{\$40}{(1 + r_d)^t}$$

where $r_d = (1.19)^{60/365} - 1 \approx 2.9\%$ per 60-day period.

Default loss (discounted): $-\text{LGD} \times \text{loan}_i \times (1+r_d)^{-t}$

### 6.3 Results

| Tier | Avg Expected NPV | Avg Default Rate | Avg Increases |
|------|:----------------:|:----------------:|:-------------:|
| Prime | −$61 | 7.3% | 2.45 |
| Near-Prime | −$336 | 19.8% | 1.80 |
| Subprime | −$470 | 24.5% | 0.82 |

The **negative average NPV** even for Prime reflects the portfolio average including ineligible customers and those with full capacity. The **374 selected Prime customers** all have positive expected NPV.

---

## 7. Optimization

### 7.1 Problem Formulation

$$\max_{x_i \in \{0,1\}} \sum_{i} x_i \cdot \mathbb{E}[\text{NPV}_i]$$

$$\text{s.t.} \quad \sum_i x_i \cdot e_i \leq C_{\text{cap}} = \$24.77\text{M}$$
$$\mathbb{E}[\text{DR} \mid \text{selected}] \leq 15\%$$
$$\text{eligible}_i = 1, \quad \text{remaining\_capacity}_i > 0, \quad \mathbb{E}[\text{NPV}_i] > 0$$

### 7.2 Solution Method

This is a **0-1 knapsack problem** (NP-hard in general). We solve the LP relaxation via greedy efficiency ratio:

$$\text{efficiency}_i = \frac{\mathbb{E}[\text{NPV}_i]}{e_i}$$

For the fractional knapsack, the greedy algorithm is **optimal**. For the integer knapsack with our constraints, it produces a high-quality feasible solution in O(n log n) time.

### 7.3 Results

- **374 customers selected** from sample (→ ~2,243 scaled)
- **All selected are Prime tier** — confirming that only low-risk, high-engagement customers generate positive NPV under the $40/increase model
- **Portfolio default rate: 8.2%** — well within the 15% regulatory cap
- **Capital utilized: 4.6%** — significant headroom; the binding constraint is the 15% default cap, not capital

---

## 8. Scenario Analysis

| Scenario | NPV | vs. Baseline |
|----------|-----|-------------|
| Baseline 2023 | $84,006 | — |
| High Inflation +3% | $67,919 | −19.1% |
| Recession +5% Unemp | $49,866 | −40.6% |
| CBK Rate Cut −2% | $91,999 | +9.5% |
| Optimistic Growth | $99,308 | +18.2% |

**Conclusion:** The strategy remains profitable across all scenarios. Even under severe recession conditions, the NPV stays positive — demonstrating strategic robustness.

---

## 9. Additional Assumptions

1. **LGD = 80%** (20% recovery): Conservative estimate for unsecured consumer lending in emerging markets. In practice, M-Pesa lockdown or salary deduction could reduce this to 40–60%.
2. **No partial increases**: Each increase is binary (granted or not). A continuous increase amount model would be a natural extension.
3. **Independent customers**: Correlations between customers (e.g., shared employer, regional economic shock) are not modelled. A copula model would capture systemic risk.
4. **Static tier assignment**: Risk tier is assigned once at analysis time. In production, it should be reassessed before each 60-day eligibility window.
5. **Homogeneous $40 profit**: In reality, profit may vary by loan amount, duration, and interest rate. A customer-specific profit model would improve the optimization.

---

## 10. Operationalization Recommendations

### Immediate (0–3 months)
1. **Deploy Prime auto-approval pipeline**: The 7,520 Prime customers should receive automated pre-approval messages via USSD/SMS.
2. **Freeze Subprime increases**: Redirect these customers to a repayment improvement program with milestone-based incentives.

### Medium-term (3–12 months)
3. **Implement dynamic risk scoring**: Monthly re-scoring using payment data; automate tier transitions.
4. **Pilot behavioral nudges**: Test personalized offer messaging (time-limited, transparent terms) to improve uptake rates.
5. **Reduce LGD**: Partner with mobile money providers to implement automatic payment deduction as implicit collateral.

### Long-term (12+ months)
6. **Reinforcement learning policy**: Train a contextual multi-armed bandit using customer features + outcomes to continuously optimize increase timing and amount.
7. **Macroeconomic integration**: Ingest real-time CBK rate, CPI, and unemployment data to automatically adjust the macro demand suppression factor.
8. **Portfolio monitoring dashboard**: Real-time tracking of default rates, NPV realization, and tier migration across the customer base.
