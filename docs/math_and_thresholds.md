# Math & Thresholds (Rule Engine)

This project uses **rule-based** bias detection (fast + deterministic) to satisfy the hackathon requirement of detecting at least:
- Overtrading
- Loss aversion
- Revenge trading

All scores are clamped to **[0, 100]** and mapped to levels:
- **LOW**: score < 45  
- **MEDIUM**: 45 ≤ score < 75  
- **HIGH**: score ≥ 75  

## Common features

Given each trade *i*:

- **Notional**:  
  \[
  notional_i = quantity_i \times entry\_price_i
  \]

- **Risk as % of balance** (proxy):  
  \[
  risk\_pct\_balance_i = \frac{|pnl_i|}{max(balance_i, \epsilon)}
  \]
  with \(\epsilon = 1e{-9}\) to avoid divide-by-zero.

- **Time delta** (minutes) between consecutive trades:  
  \[
  \Delta t_i = \frac{timestamp_i - timestamp_{i-1}}{60}
  \]

## 1) Overtrading

Signals computed on the dataset:

1. **Trades per day**
- Group by day: \(tpd_d\)
- Mean trades/day: \(\overline{tpd}\)
- 90th percentile: \(tpd_{0.9}\)
- Baseline cadence: \(TPD_{base} = 1000\)

Define:
\[
tpd\_ratio = \frac{\overline{tpd}}{TPD_{base}}
\]

Score component:
\[
tpd\_score = clamp\left( (tpd\_ratio - 1) \times 55,\; 0,\; 55 \right)
\]

2. **Hourly clustering**
- Group by hour: \(tph_h\)
- Max trades in an hour: \(max\_tph\)
- 90th percentile: \(tph_{0.9}\)
- Baseline burst: \(TPH_{base} = 50\)

\[
tph\_ratio = \frac{max\_tph}{TPH_{base}}
\]

\[
tph\_score = clamp\left( (tph\_ratio - 1) \times 30,\; 0,\; 30 \right)
\]

3. **Rapid switching rate**
Switching = (asset changes OR side changes) AND \(\Delta t \le 15\) minutes.

\[
switching\_rate = mean( switching\_event_i )
\]
\[
switch\_score = clamp\left( (switching\_rate - 0.95) \times 100 \times 0.5,\; 0,\; 5 \right)
\]

4. **Chasing after big move**
Define a “big move” as \(zscore(|pnl|) > 1.5\).  
Then “after big” = previous trade is big move AND \(\Delta t \le 30\) minutes.

\[
after\_big\_rate = mean(after\_big\_event_i)
\]
\[
chase\_score = clamp\left( (after\_big\_rate - 0.10) \times 100 \times 0.5,\; 0,\; 10 \right)
\]

**Final overtrading score**
\[
score = clamp(tpd\_score + tph\_score + switch\_score + chase\_score)
\]

## 2) Loss aversion

Let:
- \(avg\_win = mean(pnl \mid pnl>0)\)
- \(avg\_loss = mean(pnl \mid pnl\le 0)\) (negative)
- \(|avg\_loss| = abs(avg\_loss)\)
- \(win\_rate = mean( pnl>0 )\)

1. **Loss magnitude vs win magnitude**
\[
mag\_ratio = \frac{|avg\_loss|}{avg\_win} \quad (\text{if } avg\_win>0)
\]
\[
mag\_score = clamp( (mag\_ratio - 1) \times 35 )
\]

2. **Payoff (reward:risk proxy)**
\[
payoff = \frac{avg\_win}{|avg\_loss|}
\]
If payoff < 1:
\[
payoff\_score = clamp( (1 - payoff) \times 35 )
\]

3. **Holding winners shorter than losers** (proxy)
We approximate holding time by the time delta between consecutive trades **on the same asset**:
- median \(dt\_win\) for winners
- median \(dt\_loss\) for losers

\[
dt\_ratio = \frac{dt\_loss}{dt\_win}
\]
\[
dt\_score = clamp( (dt\_ratio - 1) \times 20 )
\]

4. **Profit factor**
\[
profit\_factor = \frac{\sum pnl_{wins}}{|\sum pnl_{losses}|}
\]
If profit_factor < 1.2:
\[
pf\_score = clamp( (1.2 - profit\_factor) \times 20 )
\]

**Final loss aversion score**
\[
score = clamp(mag\_score + payoff\_score + dt\_score + pf\_score)
\]

## 3) Revenge trading

1. **Risk increase after a loss**
Let:
- \(risk\_after\_loss = mean(risk\_pct\_balance \mid pnl_{prev}<0)\)
- \(risk\_after\_nonloss = mean(risk\_pct\_balance \mid pnl_{prev}\ge 0)\)

\[
risk\_ratio = \frac{risk\_after\_loss}{risk\_after\_nonloss}
\]
\[
risk\_score = clamp( (risk\_ratio - 1) \times 45 )
\]

2. **Size (notional) increase after loss streak ≥ 2**
\[
notional\_ratio = \frac{mean(notional \mid loss\_streak_{prev}\ge2)}{mean(notional \mid loss\_streak_{prev}<2)}
\]
\[
size\_score = clamp( (notional\_ratio - 1) \times 35 )
\]

3. **Too-fast re-entry after loss**
\[
too\_fast\_rate = mean( pnl_{prev}<0 \;\wedge\; \Delta t \le 10 )
\]
\[
fast\_score = clamp( too\_fast\_rate \times 100 \times 0.30 )
\]

**Final revenge trading score**
\[
score = clamp(risk\_score + size\_score + fast\_score)
\]

## Overall score

Weighted average (chosen for interpretability):
\[
overall = clamp(0.35\,score_{over} + 0.35\,score_{loss} + 0.30\,score_{rev})
\]


### Parser note
The provided hackathon CSVs use `profit_loss`; the parser maps it to `pnl` automatically.
