import numpy as np
from scipy.stats import norm

# Black-Scholes equation for the price of a call option
def black_scholes(S, K, r, sigma, t):
  d1 = (np.log(S / K) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
  d2 = d1 - sigma * np.sqrt(t)
  call = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
  return call

# parameters
S = 100 # underlying asset price
K = 105 # strike price
r = 0.05 # risk-free interest rate
sigma = 0.2 # volatility
t = 1 # time to maturity

# calculate the price of the liquidation option
call_price = black_scholes(S, K, r, sigma, t)

# print the result




