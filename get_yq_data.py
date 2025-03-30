import yahooquery as yq
from yahooquery import Ticker

aapl = Ticker('xle')

aapl.summary_detail

test=aapl.history(start='1990-01-01', adj_ohlc=True)

plt.plot(test.index.levels[1],test.close.values)