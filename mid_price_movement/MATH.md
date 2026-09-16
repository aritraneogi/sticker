# Mathematical Formulations

The document provides the collection of all metric definitions, learning dynamics, optimization updates, and evaluation metrics used.

---

## Contents

1. [Core Microstructure & Evaluation](#1-core-microstructure--evaluation)
2. [Feature Engineering](#2-feature-engineering)
3. [Baseline Architectures & Update Rules](#3-baseline-architectures--update-rules)
4. [Directional Accuracy & Alpha](#4-directional-accuracy--alpha)
5. [Theoretical Market Making & Inventory Control](#5-theoretical-market-making--inventory-control)
6. [Continual Regime Adaptation](#6-continual-regime-adaptation)
7. [Theoretical Market Taking & Friction](#7-theoretical-market-taking--friction)
8. [Queue Jumping](#8-queue-jumping)
9. [Latency Arbitrage & Alpha Decay](#9-latency-arbitrage--alpha-decay)

---

## 1. Core Microstructure & Evaluation

### 1.1 Level-4 Order Book State
At any discrete trade tick $t \in \{1, 2, \dots, N\}$, the limit order book (LOB) state is defined by the top-$K$ price levels on both sides:
$$\mathcal{L}_t = \left\{ \left(P_{t, k}^b, V_{t, k}^b\right), \left(P_{t, k}^a, V_{t, k}^a\right) \right\}_{k=1}^K$$
where $P_{t, k}^b$ and $V_{t, k}^b$ denote the bid price and volume at level $k$, and $P_{t, k}^a$ and $V_{t, k}^a$ denote the ask price and volume at level $k$.

The top-of-book ($k=1$) defines the best bid $P_{t, 1}^b = \text{BestBid}_t$ and best ask $P_{t, 1}^a = \text{BestAsk}_t$.

### 1.2 Mid-Price & Spread
The mid-price $M_t$ and bid-ask spread $S_t$ are defined as:
$$M_t = \frac{P_{t, 1}^a + P_{t, 1}^b}{2}$$
$$S_t = P_{t, 1}^a - P_{t, 1}^b$$

### 1.3 Causal Horizon Labels
For a forward trade horizon $h \in \{1, 2, 3, 5, 10, 20, 50, 100, 200, 500\}$ ticks, the forward mid-price return label $y_{t, h} \in \{-1, 0, +1\}$ is defined as:
$$\Delta M_{t, h} = M_{t+h} - M_t$$
$$y_{t, h} = \operatorname{sign}(\Delta M_{t, h}) = \begin{cases} +1, & \text{if } \Delta M_{t, h} > 0 \quad (\text{Price Up}) \\ -1, & \text{if } \Delta M_{t, h} < 0 \quad (\text{Price Down}) \\ 0, & \text{if } \Delta M_{t, h} = 0 \quad (\text{Price Flat}) \end{cases}$$

### 1.4 Prequential Protocol
At tick $t$:
1. Model receives features $x_t \in \mathbb{R}^D$ constructed strictly from $\mathcal{L}_t$ and historical trades $\tau \le t$.
2. Model emits continuous prediction score $\hat{y}_{t, h} \in \mathbb{R}$.
3. Prediction $\hat{y}_{t, h}$ is enqueued in a FIFO buffer $\mathcal{Q}_h$.
4. When tick $t+h$ arrives, label $y_{t, h}$ resolves; prediction $\hat{y}_{t, h}$ is dequeued and evaluated.
5. Online parameter update occurs using resolved pair $(x_t, y_{t, h})$.

$$\text{Information Set: } \mathcal{F}_t = \sigma\left( \{x_\tau\}_{\tau=1}^t, \{y_{\tau, h}\}_{\tau=1}^{t-h} \right)$$

---

## 2. Feature Engineering

### 2.1 Order Flow Imbalance (OFI)
Let $\Delta V_{t, k}^b$ and $\Delta V_{t, k}^a$ denote the net changes in volume at price level $k$:
$$\Delta W_{t, k}^b = \begin{cases} V_{t, k}^b, & \text{if } P_{t, k}^b > P_{t-1, k}^b \\ V_{t, k}^b - V_{t-1, k}^b, & \text{if } P_{t, k}^b = P_{t-1, k}^b \\ 0, & \text{if } P_{t, k}^b < P_{t-1, k}^b \end{cases}$$

$$\Delta W_{t, k}^a = \begin{cases} 0, & \text{if } P_{t, k}^a > P_{t-1, k}^a \\ V_{t, k}^a - V_{t-1, k}^a, & \text{if } P_{t, k}^a = P_{t-1, k}^a \\ V_{t, k}^a, & \text{if } P_{t, k}^a < P_{t-1, k}^a \end{cases}$$

The level-$k$ Order Flow Imbalance $OFI_{t, k}$ is:
$$OFI_{t, k} = \Delta W_{t, k}^b - \Delta W_{t, k}^a$$

Multi-scale exponentially decaying OFI over decay factor $\alpha \in (0, 1)$:
$$OFI_t^{(\alpha)} = \alpha \cdot OFI_{t-1}^{(\alpha)} + (1 - \alpha) \cdot OFI_{t, 1}$$

### 2.2 Microprice & Volume Imbalance
The Level-1 volume imbalance ratio $I_t \in [-1, 1]$ is:
$$I_t = \frac{V_{t, 1}^b - V_{t, 1}^a}{V_{t, 1}^b + V_{t, 1}^a}$$

The Level-1 Microprice $P_t^{\text{micro}}$ is:
$$P_t^{\text{micro}} = \frac{V_{t, 1}^b P_{t, 1}^a + V_{t, 1}^a P_{t, 1}^b}{V_{t, 1}^b + V_{t, 1}^a} = M_t + \frac{S_t}{2} I_t$$

Multi-level depth weighted volume imbalance:
$$I_t^{(K)} = \frac{\sum_{k=1}^K w_k \left(V_{t, k}^b - V_{t, k}^a\right)}{\sum_{k=1}^K w_k \left(V_{t, k}^b + V_{t, k}^a\right)}, \quad w_k = \frac{1}{k}$$

### 2.3 Volume-Weighted Average Price (VWAP) Deviation
For execution window $W$:
$$\text{VWAP}_{t, W} = \frac{\sum_{i=0}^{W-1} P_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{W-1} V_{t-i}}$$
$$\Delta \text{VWAP}_{t, W} = \frac{M_t - \text{VWAP}_{t, W}}{S_t + \epsilon}$$

### 2.4 Online Normalization (Welford's Algorithm)
Features are standardized online without lookahead using Welford recurrence:
$$\mu_t = \mu_{t-1} + \frac{x_t - \mu_{t-1}}{t}$$
$$M_{2, t} = M_{2, t-1} + (x_t - \mu_{t-1})(x_t - \mu_t)$$
$$\sigma_t^2 = \frac{M_{2, t}}{t - 1}$$
$$\tilde{x}_t = \frac{x_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}$$

---

## 3. Baseline Architectures & Update Rules

### 3.1 TLOB

#### Self-Attention Mechanism
Input sequence $X \in \mathbb{R}^{L \times D}$ where $L=128$, projected into Queries, Keys, Values:
$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V, \quad W^Q, W^K, W^V \in \mathbb{R}^{D \times d_{\text{model}}}$$
$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

#### Multi-Head Formulation
$$\operatorname{MHA}(X) = \left[\operatorname{head}_1 \mathbin{\Vert} \dots \mathbin{\Vert} \operatorname{head}_{n_h}\right] W^O$$
$$\operatorname{head}_i = \operatorname{Attention}\left(X W_i^Q, X W_i^K, X W_i^V\right)$$

#### AdamW Optimization Step
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

---

### 3.2 DeepLOB

#### 2D Spatial-Temporal Convolution
For receptive field kernel $K \in \mathbb{R}^{f_h \times f_w}$:
$$C_{i, j} = \operatorname{LeakyReLU}\left(\sum_{m} \sum_{n} X_{i+m, j+n} K_{m, n} + b\right)$$

#### LSTM Recurrent Gate Transitions
$$f_t = \sigma\left(W_f x_t + U_f h_{t-1} + b_f\right) \quad (\text{Forget Gate})$$
$$i_t = \sigma\left(W_i x_t + U_i h_{t-1} + b_i\right) \quad (\text{Input Gate})$$
$$\tilde{c}_t = \tanh\left(W_c x_t + U_c h_{t-1} + b_c\right) \quad (\text{Candidate Cell})$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad (\text{Cell State})$$
$$o_t = \sigma\left(W_o x_t + U_o h_{t-1} + b_o\right) \quad (\text{Output Gate})$$
$$h_t = o_t \odot \tanh(c_t) \quad (\text{Hidden State})$$

---

### 3.3 Mamba

#### Continuous-Time State Space Equation
$$h'(t) = A h(t) + B x(t)$$
$$y(t) = C h(t)$$

#### Zero-Order Hold (ZOH) Discretization
Given input-dependent step size $\Delta_t = \operatorname{softplus}(\operatorname{Linear}_\Delta(x_t))$:
$$\bar{A}_t = \exp\left(\Delta_t A\right)$$
$$\bar{B}_t = \left(\Delta_t A\right)^{-1} \left(\exp(\Delta_t A) - I\right) \cdot \left(\Delta_t B_t\right) \approx \Delta_t B_t$$

#### Selective Recurrence
$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$$
$$y_t = C_t h_t$$
where $B_t = W_B x_t$, $C_t = W_C x_t$.

---

### 3.4 Kalman Filter

#### Kinematic State Formulation
$$x_t = \begin{bmatrix} M_t \\ v_t \end{bmatrix}, \quad F = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \quad H = \begin{bmatrix} 1 & 0 \end{bmatrix}$$

#### Riccati Prediction & Update Equations
$$\hat{x}_{t|t-1} = F \hat{x}_{t-1|t-1}$$
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
$$K_t = P_{t|t-1} H^T \left(H P_{t|t-1} H^T + R\right)^{-1}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \left(z_t - H \hat{x}_{t|t-1}\right)$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

---

### 3.5 ARIMA

#### AR(p) Representation
$$y_t = \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t = \phi^T \mathbf{y}_{t-1} + \epsilon_t$$

#### RLS Gain & Covariance Update
$$K_t = \frac{P_{t-1} \mathbf{y}_{t-1}}{\lambda + \mathbf{y}_{t-1}^T P_{t-1} \mathbf{y}_{t-1}}$$
$$\hat{\phi}_t = \hat{\phi}_{t-1} + K_t \left(y_t - \hat{\phi}_{t-1}^T \mathbf{y}_{t-1}\right)$$
$$P_t = \frac{1}{\lambda} \left( P_{t-1} - K_t \mathbf{y}_{t-1}^T P_{t-1} \right)$$
where $\lambda \in (0, 1]$ is the exponential forgetting factor.

---

### 3.6 GARCH(1,1)

#### Conditional Variance Dynamics
$$r_t = \ln\left(\frac{M_t}{M_{t-1}}\right) = \sigma_t z_t, \quad z_t \sim \mathcal{N}(0, 1)$$
$$\sigma_t^2 = \omega + \alpha r_{t-1}^2 + \beta \sigma_{t-1}^2$$
subject to constraints $\omega > 0$, $\alpha \ge 0$, $\beta \ge 0$, $\alpha + \beta < 1$.

---

## 4. Directional Accuracy & Alpha

### 4.1 Non-Zero Directional Accuracy ($\text{DA}_{nz}$)
Let $\mathcal{T}_{nz} = \{t \mid y_{t, h} \neq 0\}$ and $N_{nz} = |\mathcal{T}_{nz}|$:
$$\text{DA}_{nz}(h) = \frac{1}{N_{nz}} \sum_{t \in \mathcal{T}_{nz}} \mathbb{I}\left(\operatorname{sign}(\hat{y}_{t, h}) = y_{t, h}\right)$$

### 4.2 All-Tick Directional Accuracy ($\text{DA}_{all}$)
$$\text{DA}_{all}(h) = \frac{1}{N} \sum_{t=1}^N \mathbb{I}\left(\operatorname{sign}(\hat{y}_{t, h}) = y_{t, h}\right)$$

### 4.3 Matthews Correlation Coefficient (MCC)
$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$
where:
* $TP = \sum \mathbb{I}(y_t=+1, \hat{y}_t>0)$
* $TN = \sum \mathbb{I}(y_t=-1, \hat{y}_t<0)$
* $FP = \sum \mathbb{I}(y_t=-1, \hat{y}_t>0)$
* $FN = \sum \mathbb{I}(y_t=+1, \hat{y}_t<0)$

### 4.4 Macro & Weighted $F_1$-Score
$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
$$F_{1, c} = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
$$F_{1, \text{macro}} = \frac{1}{2}\left(F_{1, +1} + F_{1, -1}\right), \quad F_{1, \text{weighted}} = \frac{N_+ F_{1, +1} + N_- F_{1, -1}}{N_+ + N_-}$$

### 4.5 Signal Edge & Profit Factor
$$\text{Edge}_{\text{bps}} = \frac{1}{N_{nz}} \sum_{t \in \mathcal{T}_{nz}} \operatorname{sign}(\hat{y}_{t, h}) \cdot \left(\frac{M_{t+h} - M_t}{M_t}\right) \times 10^4$$
$$\text{Profit Factor} = \frac{\sum_{t} \max\left(0, \operatorname{sign}(\hat{y}_{t, h}) \Delta M_{t, h}\right)}{\sum_{t} \max\left(0, -\operatorname{sign}(\hat{y}_{t, h}) \Delta M_{t, h}\right)}$$

### 4.6 Tick-Annualized Sharpe & Sortino
Let $r_t = \operatorname{sign}(\hat{y}_{t, h}) \frac{\Delta M_{t, h}}{M_t}$ and $K_{\text{year}} = \frac{N_{\text{total}}}{31} \times 365$:
$$\text{Sharpe}_{\text{tick}} = \frac{\mathbb{E}[r_t]}{\sigma(r_t)} \sqrt{K_{\text{year}}}$$
$$\text{Sortino}_{\text{tick}} = \frac{\mathbb{E}[r_t]}{\sqrt{\mathbb{E}\left[\min(0, r_t - \tau)^2\right]}} \sqrt{K_{\text{year}}}$$

---

## 5. Theoretical Market Making & Inventory Control

### 5.1 Signal-Guided Quote Filtering Execution Model
At tick $t$, the market maker posts passive two-sided quotes at fixed half-spread $\delta_{\text{half}} = 0.5\text{ bps}$ and collects maker rebate $r_{\text{maker}} = 1.0\text{ bps}$.
When directional signal $\hat{y}_t \in \{-1, 0, +1\}$ is emitted with logit score $z_t$, quote activation follows:
$$\text{Active}_t = \mathbb{I}(|z_t| > \tau_{\text{logit}})$$
The fill acceptance indicator $\mathcal{M}_{\text{fill}, t} \in \{0, 1\}$ against incoming trade side $\eta_t \in \{+1 (\text{Buyer Aggressor}), -1 (\text{Seller Aggressor})\}$ is:
$$\mathcal{M}_{\text{fill}, t} = \mathbb{I}\left( \neg \text{Active}_t \lor (\hat{y}_t = 0) \lor (\hat{y}_t = +1 \land \eta_t = -1) \lor (\hat{y}_t = -1 \land \eta_t = +1) \right)$$

### 5.2 Asymmetric Quote Suppression
* **Neutral ($\hat{y}_t = 0$ or Inactive)**: Both Bid and Ask quotes are active; all incoming trades are accepted.
* **Predict Up ($\hat{y}_t = +1$)**: Ask quote is suppressed to avoid adverse selection against buyers; only Bid fills ($\eta_t = -1$) are accepted.
* **Predict Down ($\hat{y}_t = -1$)**: Bid quote is suppressed to avoid adverse selection against sellers; only Ask fills ($\eta_t = +1$) are accepted.

### 5.3 Inventory Half-Life ($\tau_{1/2}$)
Inventory process $q_t$ is modeled as a continuous Ornstein-Uhlenbeck (OU) mean-reversion process:
$$dq_t = -\theta q_t dt + \sigma_q dW_t$$
Estimated discretely via AR(1) regression $q_t = \rho q_{t-1} + \epsilon_t$:
$$\theta = -\ln(\rho) \implies \tau_{1/2} = \frac{\ln(2)}{\theta} = \frac{\ln(2)}{-\ln(\rho)} \quad (\text{in ticks})$$

### 5.4 Volume-Synchronized Probability of Toxicity (VPIN)
Order flow is partitioned into volume buckets of size $V_{\text{bucket}} = 200\text{ trades}$:
$$\text{VPIN} = \frac{1}{N \cdot V_{\text{bucket}}} \sum_{\tau=1}^N \left| V_\tau^B - V_\tau^S \right|$$
where $V_\tau^B$ and $V_\tau^S$ represent buy and sell volume in bucket $\tau$.

### 5.5 Adverse Selection & Realized Spread
For fill price $P_{\text{fill}}$ and trade direction $\eta \in \{+1 (\text{Buy}), -1 (\text{Sell})\}$:
$$\text{Effective Half-Spread: } \text{EHS}_t = \eta_t \left(P_{\text{fill}, t} - M_t\right)$$
$$\text{Price Impact (Adverse Selection at } k \text{ ticks}): \text{AdvSel}_{t, k} = \eta_t \left(M_{t+k} - M_t\right)$$
$$\text{Realized Half-Spread: } \text{RHS}_{t, k} = \text{EHS}_t - \text{AdvSel}_{t, k} = \eta_t \left(P_{\text{fill}, t} - M_{t+k}\right)$$

### 5.6 Market Making PnL Attribution
$$\text{Gross PnL} = \text{Spread Earned} + \text{Maker Rebates} + \text{Adverse Selection Avoidance Gain}$$
$$\text{Net PnL} = \text{Spread Earned} + \text{Maker Rebates} - \text{Adverse Selection Incurred}$$
$$\text{Attribution Fraction } \phi_i = \frac{\text{Component}_i}{\text{Gross PnL}} \times 100\%$$

---

## 6. Continual Regime Adaptation

### 6.1 Volatility & Trend Estimators
For rolling window $W = 10,000\text{ ticks}$:
$$\sigma_t^{(W)} = \sqrt{\frac{1}{W} \sum_{i=0}^{W-1} \left(r_{t-i} - \bar{r}\right)^2} \times 10^4 \quad (\text{bps})$$
$$\mu_t^{(W)} = \frac{1}{W} \sum_{i=0}^{W-1} r_{t-i} \times 10^4 \quad (\text{bps})$$

### 6.2 Gaussian Mixture Model (GMM) 3-State Partitioning
$$\mathcal{R}_t = \begin{cases} \text{State 0: Low-Vol Consolidation}, & \text{if } \sigma_t^{(W)} < 15\text{ bps} \\ \text{State 1: Moderate Trend}, & \text{if } 15\text{ bps} \le \sigma_t^{(W)} \le 45\text{ bps} \\ \text{State 2: High-Vol Shock}, & \text{if } \sigma_t^{(W)} > 45\text{ bps} \end{cases}$$

### 6.3 Backward Transfer ($BWT$)
Let $\text{DA}_{\text{Late}}(\mathcal{R}_0)$ denote accuracy during the final recurrence of State 0, and $\text{DA}_{\text{Early}}(\mathcal{R}_0)$ denote accuracy during the first encounter:
$$BWT = \text{DA}_{\text{Late}}(\mathcal{R}_0) - \text{DA}_{\text{Early}}(\mathcal{R}_0)$$
* $BWT > 0 \implies$ Positive backward transfer (knowledge consolidation).
* $BWT < 0 \implies$ Catastrophic forgetting.

### 6.4 Plasticity Speed (Post-Shift Recovery)
Accuracy measured over the first $\Delta t = 500\text{ ticks}$ following regime transition $\tau_c$:
$$\text{DA}_{\text{post-shift}} = \frac{1}{500} \sum_{t = \tau_c}^{\tau_c + 500} \mathbb{I}\left(\operatorname{sign}(\hat{y}_t) = y_t\right)$$

### 6.5 Shannon Entropy of Predictions
To quantify representational collapse vs. healthy diversity across discrete state outputs $p_i = P(\operatorname{sign}(\hat{y}) = i)$:
$$H(\hat{y}) = -\sum_{i \in \{-1, +1\}} p_i \log_2(p_i) \in [0, 1]$$

### 6.6 Continual Adaptation Score ($S_{\text{Continual}}$)
$$S_{\text{Continual}} = \frac{\text{DA}_{\text{shock}} \times \left(1 + \frac{BWT}{100}\right) \times \left(\frac{\text{DA}_{\text{post-shift}}}{\text{DA}_{\text{steady}}}\right) \times H(\hat{y})}{\mathcal{P}_{\text{latency}}}$$
where $\mathcal{P}_{\text{latency}} = \max\left(1.0, \frac{\text{Latency}_{\mu s}}{20.0}\right)$.

---

## 7. Theoretical Market Taking & Friction

### 7.1 Exchange Execution Friction Calibration
* Taker Fee: $f_{\text{taker}} = 2.5\text{ bps / side}$ ($0.025\%$)
* Maker Rebate: $r_{\text{maker}} = 1.0\text{ bps / side}$ ($0.010\%$ rebate)
* Half Spread: $\delta_{\text{half}} = 0.5\text{ bps / side}$
* Execution Slippage: $\xi_{\text{slip}} = 0.5\text{ bps / side}$

### 7.2 Friction Scenarios (Round-Trip)
1. **Optimistic (Maker-Maker Net)**: $C_{\text{RT}} = 2 \times (\delta_{\text{half}} - r_{\text{maker}}) = 2 \times (0.5 - 1.0) = -1.0\text{ bps} \implies \text{Bound to } 4.0\text{ bps}$
2. **Realistic (Taker-Taker Baseline)**: $C_{\text{RT}} = 2 \times (f_{\text{taker}} + \delta_{\text{half}} + \xi_{\text{slip}}) = 2 \times (2.5 + 0.5 + 0.5) = 7.0\text{ bps}$
3. **Pessimistic (Adverse Slippage)**: $C_{\text{RT}} = 18.0\text{ bps}$

### 7.3 Net Trade PnL Dynamics
For trade position taken at tick $t$ and closed at $t+h$ with notional $V_{\text{USD}} = \$100,000$:
$$R_{\text{gross}, t} = \operatorname{sign}(\hat{y}_t) \cdot \left(\frac{M_{t+h} - M_t}{M_t}\right)$$
$$R_{\text{net}, t} = R_{\text{gross}, t} - \frac{C_{\text{RT}}}{10^4}$$
$$\text{PnL}_{\text{USD}, t} = V_{\text{USD}} \cdot R_{\text{net}, t}$$

### 7.4 Critical Breakeven Cost ($C_{\text{crit}}$)
The maximum allowable transaction friction before net alpha collapses to zero:
$$C_{\text{crit}} = \mathbb{E}\left[ R_{\text{gross}, t} \right] \times 10^4 \quad (\text{bps})$$

---

## 8. Queue Jumping

### 8.1 Queue Priority Fill Probability ($P_{\text{fill}}$)
For incoming trade size $V_{\text{trade}}$ and queue volume ahead $V_{\text{queue}}$ with price improvement factor $\phi_{\text{jump}} = 0.70$:
$$V_{\text{eff\_pos}} = V_{\text{queue}} \cdot (1 - \phi_{\text{jump}})$$
$$P_{\text{fill\_raw}} = \min\left(1.0, \frac{V_{\text{trade}}}{V_{\text{eff\_pos}}}\right)$$
$$P_{\text{fill}} = P_{\text{fill\_raw}} \cdot \exp(-\lambda_{\text{arrival}} \cdot h_{\text{lag}})$$
where $\lambda_{\text{arrival}} = 0.18$ and $h_{\text{lag}} = \operatorname{round}(p50 / 22\,\mu\text{s})$.

### 8.2 Winner's Curse Penalty ($\text{WC}_k$)
The adverse selection conditioned on receiving a passive fill at the bid at tick $t$:
$$\text{WC}_k = \mathbb{E}\left[ M_t - M_{t+k} \;\middle|\; \text{Passive Fill at Bid at } t \right]$$

### 8.3 Flow Purity ($\% \text{Safe}$)
$$\text{Flow Purity} = 100\% - \text{Adverse Selection Rate}_{5\text{tk}} = 100\% \times \left(1 - \frac{\sum \mathbb{I}(\text{AdvSel}_{t, 5} > 0)}{N_{\text{fills}}}\right)$$

### 8.4 Capital & Risk-Adjusted Institutional Quality Score ($Q_{\text{Risk}}$)
$$Q_{\text{Risk}} = \frac{\text{DA}_{\%} \times \text{Flow Purity}_{\%} \times \text{Edge}_{\text{bps/fill}}}{\sigma_{\text{Inv}} \times \mathcal{P}_{\text{latency}}}$$
where:
* $\sigma_{\text{Inv}} = \text{Standard deviation of open BTC inventory balance}$.
* $\text{Edge}_{\text{bps/fill}} = \frac{\text{Total Net PnL (bps)}}{N_{\text{executed fills}}}$.
* $\mathcal{P}_{\text{latency}} = \max\left(1.0, \frac{\text{p50 Latency}_{\text{ms}}}{0.020}\right)$.

---

## 9. Latency Arbitrage & Alpha Decay

### 9.1 Empirical Alpha Decay Function
Microstructure predictive signal edge decays exponentially over latency delay $\delta$:
$$\alpha(\delta) = \alpha_0 \cdot \exp\left(-\lambda_{\text{decay}} \cdot \delta\right)$$
where $\lambda_{\text{decay}} = \frac{\ln(2)}{\tau_{1/2}^{\text{alpha}}}$ and $\tau_{1/2}^{\text{alpha}}$ is the empirical alpha half-life.

### 9.2 Hardware Latency Penalty Function ($\mathcal{P}_{\text{lat}}$)
Models incurring execution latency $\tau$ receive an empirical throughput degradation factor:
$$\mathcal{P}_{\text{lat}}(\tau) = \begin{cases} 1.0, & \text{if } \tau \le 20\,\mu\text{s} \\ \exp\left(-\kappa_{\text{lat}} (\tau - 20\,\mu\text{s})\right), & \text{if } \tau > 20\,\mu\text{s} \end{cases}$$

### 9.3 Dual-Execution Latency Arbitrage Models

#### 1. Single-Venue Breakeven Latency Arbitrage
For signal-driven quote sniping on a single venue with round-trip fee friction $C_{\text{RT}}$:
$$\text{Gross}_{\text{trade}} = \frac{1}{N_{\text{trades}}} \sum_{i=1}^{N_{\text{trades}}} \operatorname{sign}(\hat{y}_{t_i}) \cdot \left(\frac{M_{t_i+h} - M_{t_i}}{M_{t_i}}\right) \times 10^4 \quad (\text{bps})$$
$$\text{Breakeven Hurdle Ratio} = \frac{C_{\text{RT}}}{\text{Gross}_{\text{trade}}}$$
where $C_{\text{RT}} = 7.0\text{ bps}$ for Taker-Taker and $C_{\text{RT}} = 2.5\text{ bps}$ for Taker-Maker execution.

#### 2. Passive Maker-Gated Alpha Arbitrage
$$\text{PnL}_{\text{maker}}(\tau) = P_{\text{fill}}(\tau) \cdot \left[ 2 \cdot \delta_{\text{half}} + 2 \cdot r_{\text{maker}} + \alpha(\tau) \right]$$

### 9.4 Net Alpha Capture Efficiency ($\eta_{\text{capture}}$)
$$\eta_{\text{capture}}(\tau) = \frac{\text{PnL}_{\text{net}}(\tau)}{\text{PnL}_{\text{net}}(\tau = 0)} \times 100\%$$
