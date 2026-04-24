# Future Developments in Applied Econometrics

This document outlines potential future enhancements and advanced econometric techniques that can be integrated into the current Headline CPI forecasting pipeline.

## 1. Mixed-Data Sampling (MIDAS) Models
Currently, the pipeline uses linear interpolation to align different frequencies of data (e.g., weekly Forex reserves, daily USD/INR) to a common monthly frequency.
- **Enhancement:** Implement MIDAS regressions to allow high-frequency data (like daily commodity prices or exchange rates) to directly forecast the low-frequency target (monthly CPI) without the information loss associated with temporal aggregation.

## 2. Regime-Switching Models (Markov-Switching VAR)
Inflation dynamics often behave differently during periods of crisis (e.g., COVID-19 pandemic) compared to stable economic periods.
- **Enhancement:** Transition from a constant-parameter VAR to a Markov-Switching VAR (MS-VAR). This would allow the model to automatically detect and shift between "high-inflation volatility" regimes and "stable-inflation" regimes, adjusting the coefficient matrices accordingly.

## 3. Structural Vector Autoregression (SVAR)
While the current VAR model is excellent for forecasting, it has limited capabilities for causal inference and impulse response analysis because the shocks are contemporaneously correlated.
- **Enhancement:** Apply a Structural VAR with theoretical restrictions (e.g., Cholesky decomposition or long-run restrictions via Blanchard-Quah). This will allow us to isolate orthogonal structural shocks (e.g., pure monetary policy shocks vs. supply-side commodity shocks) and trace their exact impulse response functions on CPI.

## 4. Nonlinear Cointegration and Threshold VECM (TVECM)
The Johansen cointegration test assumes a linear long-run equilibrium. However, central banks often intervene asymmetrically (e.g., defending the currency only when it depreciates past a certain threshold).
- **Enhancement:** Implement Threshold VECM to account for transaction costs or asymmetric policy responses. In TVECM, the speed of adjustment back to the long-run equilibrium depends on the magnitude of the deviation.

## 5. Dynamic Factor Models (DFM) and Machine Learning Hybrids
As we incorporate more macro variables (dozens or hundreds), traditional VAR models suffer from the "curse of dimensionality."
- **Enhancement:** Use Principal Component Analysis (PCA) to extract common latent factors from a large dataset of macroeconomic indicators (DFM-VAR). Alternatively, integrate machine learning regularization techniques (like LASSO-VAR) to shrink insignificant coefficients to zero and improve out-of-sample prediction.

## 6. Bayesian VAR (BVAR)
To prevent overfitting with the expanded Model 2 (which includes many macro controls), we can introduce prior beliefs about the parameters.
- **Enhancement:** Implement Bayesian VAR with Minnesota priors. This assumes that most macroeconomic series follow a random walk, heavily shrinking the coefficients of distant lags towards zero, leading to superior forecasting performance over traditional OLS VARs.
