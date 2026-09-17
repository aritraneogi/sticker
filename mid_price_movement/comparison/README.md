# Comparison

Core model performance metrics are available in [`core_model_performance/`](core_model_performance/).

Trading specific theoretical market-taking metrics are in [`theoretical_market_taking/`](theoretical_market_taking/) and theoretical market-making metrics reside in [`theoretical_market_making/`](theoretical_market_making/) 

Realistic trading metrics are computed for [`continual_regime_adaptation/`](continual_regime_adaptation/), [`latency_arbitrage/`](latency_arbitrage/) and [`queue_jumping/`](queue_jumping/)

Each scenario has different subset of metrics with respective sort criteria for 89 ranked models.

Primary metrics and ranking are done on last 75% split of dataset, that is 8,939,160 transactions or orders, to capture the 'learnt state' of each model.

Theoretical metrics are directly derived from core model performance metrics while realistic metrics contain additional fees and slippage considerations.

> [!NOTE]
> Same metrics might appear across different scenarios with different scalar values, refer to scenario specific documentation and scripts for derivation and computation of each.