#⛓︎ Work authored by Bernardo Kautz. Follow on LinkedIn: https://www.linkedin.com/in/bernardo-kautz/.

import warnings
import numpy as np
import yfinance as yf
import matplotlib.pyplot as pyplot
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.simplefilter(action = 'ignore', category = FutureWarning)

ticker = 'VBR' #✎ Settable parameter(s)
start = datetime.now() - relativedelta(years = 10) #✎ Adjustable parameter(s)
end = datetime.now() #✎ Adjustable parameter(s)
data = yf.download(ticker, start = start, end = end, auto_adjust = True)['Close']

log_returns = np.log(data / data.shift(1)).dropna()
initial_price = float(data.iloc[-1])
drift = float(log_returns.mean() * 252)

lambda_ = 0.94 #✎ Adjustable parameter(s)
ewma = log_returns.ewm(alpha = 1 - lambda_, adjust = False).var()
volatility = float(np.sqrt(ewma.iloc[-1] * 252))

time_horizon = 0.25 #✎ Adjustable parameter(s)
time_steps = int(252 * time_horizon)
simulations = 100000 #✎ Adjustable parameter(s)

delta_time = time_horizon / time_steps
random_matrix = np.random.standard_normal((time_steps, simulations))
price_paths = np.zeros_like(random_matrix)
price_paths[0] = initial_price

for step in range(1, time_steps):
    price_paths[step] = price_paths[step - 1] * np.exp(
        (drift - 0.5 * volatility ** 2) * delta_time + volatility * np.sqrt(delta_time) * random_matrix[step]
    )

final_prices = price_paths[-1]
expected_price = np.mean(final_prices)
expected_return = (expected_price / initial_price - 1) * 100
percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
percentile_returns = (percentiles / initial_price - 1) * 100

print('\n===| Performance Insights |===')
print(f'Ticker: {ticker}')
print(f'Current price: {initial_price:.2f}')
print(f'Expected value: {expected_price:.2f} ({expected_return:.2f}%)')
print(f'Annualized volatility*: {volatility:.4f}')
print(f'Time horizon: {time_horizon} year(s)')

print('\n===| Main Percentiles |===')
print(f'5th: {percentiles[0]:.2f} ({percentile_returns[0]:.2f}%)')
print(f'25th: {percentiles[1]:.2f} ({percentile_returns[1]:.2f}%)')
print(f'Median: {percentiles[2]:.2f} ({percentile_returns[2]:.2f}%)')
print(f'75th: {percentiles[3]:.2f} ({percentile_returns[3]:.2f}%)')
print(f'95th: {percentiles[4]:.2f} ({percentile_returns[4]:.2f}%)')

print('\n*Estimated through Exponentially Weighted Moving Average (EWMA)')

figure, (axis_trajectories, axis_histogram) = pyplot.subplots(1, 2, figsize = (12, 5), gridspec_kw = {'width_ratios': [3, 1]}, sharey = True)

axis_trajectories.plot(price_paths)
axis_trajectories.set_title(f'Geometric Brownian Motion (GBM) — {ticker}, {time_horizon} year(s)', fontsize = 12.5)
axis_trajectories.set_xlabel('Business Days', fontsize = 10)
axis_trajectories.set_ylabel('Simulated Prices', fontsize = 10)
axis_trajectories.grid(True, linestyle = ':', alpha = 0.4)
axis_trajectories.tick_params(axis = 'both', labelsize = 8.5)

axis_histogram.hist(final_prices, bins = 30, orientation = 'horizontal', color = 'slategray', edgecolor = 'white')
axis_histogram.set_title('Distribution', fontsize = 12.5)
axis_histogram.set_xlabel('Occurrences', fontsize = 10)
axis_histogram.grid(True, linestyle = ':', alpha = 0.4)
axis_histogram.tick_params(axis = 'both', labelsize = 8.5)
axis_histogram.axhspan(percentiles[0], percentiles[4], color = 'lightgray', alpha = 0.4, label = '5th–95th Percentile Interval')
axis_histogram.legend(fontsize = 8.5, loc = 'best')

pyplot.tight_layout()

pyplot.show()
