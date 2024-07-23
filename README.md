# Markowitz's Portfolio Optimization Theory and Sharpe Ratio

## Overview

Markowitz's portfolio optimization theory is a fundamental component of modern portfolio theory. It focuses on selecting a combination of assets to maximize expected return for a given level of risk, or to minimize risk for a given level of expected return. 
This repo provides an overview of Markowitz's theory and integrates the Sharpe Ratio for evaluating portfolio performance.

## 1. Diversification

Markowitz demonstrated that total portfolio risk can be reduced through diversification. This involves combining assets that do not move perfectly in sync with each other, thereby reducing overall volatility. 
The key idea is that negative fluctuations in some assets can be offset by positive fluctuations in others.

## 2. Expected Return and Risk

### Expected Return

The expected return of a portfolio is the weighted average of the expected returns of the individual assets. If ![Fórmula](https://latex.codecogs.com/svg.image?E(R_i)) represents the expected return of asset ![Fórmula](https://latex.codecogs.com/svg.image?i) and ![Fórmula](https://latex.codecogs.com/svg.image?w_i) is the weight of asset ![Fórmula](https://latex.codecogs.com/svg.image?i) in the portfolio, the expected return of the portfolio ![Fórmula](https://latex.codecogs.com/svg.image?E(R_p)) is:

![Fórmula](https://latex.codecogs.com/svg.image?E(R_p)=\sum_{i=1}^{n}w_i&space;E(R_i))

In this case we work with the logarithm of the returns. The necessary modules are imported and the data is downloaded from YahooFinance. 
The result is a two-column DataFrame, named after the corresponding share, where the time series of the adjusted close for each share has been stored.
Another DataFrame is created that stores the logarithm of the returns (it is nothing more than the share price at t+1 divided by the share price at t).

      import pandas as pd
      import numpy as np
      import yfinance as yf
      import matplotlib.pyplot as plt
      import scipy.optimize as optimize
      import sys
      
      assets = tickers = ['TSLA','^GSPC']
      
      data = pd.DataFrame()
      for t in assets:
          data[t] = yf.download(t, start='2015-1-1')['Adj Close']
      
      log_returns = np.log(1+data.pct_change())

In this case, 100,000 Monte Carlo steps are simulated. In each simulation, two weights are randomly generated and subsequently normalised. 
The expected annualised return of the portfolio is stored in port_ret, with the expected returns of each share being the average of the historical returns.

      port_returns = []
      port_vols = []
      
      for i in range (100000):
          num_assets = len(assets)
          weights = np.random.random(num_assets)
          weights /= np.sum(weights) 
          port_ret = np.sum(log_returns.mean() * weights) * 252

### Risk (Variance/Standard Deviation)

The risk of a portfolio, measured by its variance or standard deviation, is not simply the weighted average of the risks of the individual assets. It also depends on the covariances between assets. The variance of the portfolio ![Fórmula](https://latex.codecogs.com/svg.image?\sigma^2_p) is:

![Fórmula](https://latex.codecogs.com/svg.image?\sigma^2_p=\sum_{i=1}^{n}\sum_{j=1}^{n}w_i&space;w_j\sigma_{ij})

where ![Fórmula](https://latex.codecogs.com/svg.image?\sigma_{ij}) is the covariance between the returns of assets `i` and `j`.

The loop over the 100,000 experiments continues to calculate the annualised standard deviation (risk).

      port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))

Finally, we add the annualised return and annualised risk of the simulated portfolio i to the empty lists.

      port_returns.append(port_ret)
      port_vols.append(port_var)
      
## 3. Efficient Frontier

The efficient frontier represents the set of all portfolios that offer the maximum expected return for a given level of risk or the minimum risk for a given level of expected return. It is a curve in the risk-return space that shows the optimal combinations of assets.

## 4. Sharpe Ratio

The Sharpe Ratio is used to adjust a portfolio’s return for its risk. It is calculated as the difference between the expected return of the portfolio and the risk-free rate, divided by the standard deviation of the portfolio’s returns. It helps evaluate the return per unit of risk. Mathematically, it is expressed as:

![Fórmula](https://latex.codecogs.com/svg.image?\text{Sharpe&space;Ratio}=\frac{E(R_p)-R_f}{\sigma_p})

where:
- ![Fórmula](https://latex.codecogs.com/svg.image?E(R_p)) is the expected return of the portfolio.
- ![Fórmula](https://latex.codecogs.com/svg.image?R_f) is the risk-free rate (in this case assumed to be 0).
- ![Fórmula](https://latex.codecogs.com/svg.image?\sigma_p) is the standard deviation of the portfolio’s returns.

          sharpe = port_ret/port_var    
         
A higher Sharpe Ratio indicates a higher return per unit of risk, making the portfolio more attractive.
Finally, we obtain the annualised returns and annualised risks of each simulation, and all the sharpe ratios of 
these simulations. We are interested in keeping the sharpe ratio closest to 1, obviously. 

We identify the simulation with the best returns. 

      port_returns = np.array(port_returns)
      port_vols = np.array(port_vols)
      sharpe = port_returns/port_vols
      
      max_sr_vol = port_vols[sharpe.argmax()]
      max_sr_ret = port_returns[sharpe.argmax()]

## 5. Portfolio Optimization

To find the optimal portfolio, the Sharpe Ratio can be used as an optimization criterion. The portfolio that maximizes the Sharpe Ratio is known as the market portfolio in the Sharpe equilibrium model. This can be formulated as:

![Fórmula](https://latex.codecogs.com/svg.image?\max_{\mathbf{w}}\frac{\mathbf{w}^T\mathbf{E}-R_f}{\sqrt{\mathbf{w}^T\Sigma\mathbf{w}}})

where ![Fórmula](https://latex.codecogs.com/svg.image?\mathbf{w}) is the vector of asset weights, ![Fórmula](https://latex.codecogs.com/svg.image?\mathbf{E}) is the vector of expected returns, and ![Fórmula](https://latex.codecogs.com/svg.image?\Sigma) is the covariance matrix.

First the maximisation function is defined and then the constraints are defined. Once the weights that maximise the sharpe ratio are obtained, 
they are introduced into the portfolio_stats function and both the annualised return and the annualised risk of the optimal portfolio are obtained.

       def portfolio_stats(weights, log_returns):
                port_ret = np.sum(log_returns.mean() * weights) * 252
                port_var = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
                sharpe = port_ret/port_var    
                return {'Return': port_ret, 'Volatility': port_var, 'Sharpe': sharpe}
                
        def minimize_sharpe(weights, log_returns): 
            return -portfolio_stats(weights, log_returns)['Sharpe'] 
        
        constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
        bounds = tuple((0,1) for x in range(num_assets))
        initializer = num_assets * [1./num_assets,]
        
        optimal_sharpe = optimize.minimize(minimize_sharpe, initializer, method = 'SLSQP', args = (log_returns,) ,bounds = bounds, constraints = constraints)
        optimal_sharpe_weights = optimal_sharpe['x'].round(4)
        optimal_stats = portfolio_stats(optimal_sharpe_weights, log_returns)

The results conclude the same. The same efficient portfolio is arrived at by two different approaches.

## Efficient Frontier and Optimal Portfolio

![Texto Alternativo](results/efficient_frontier.png)


        
