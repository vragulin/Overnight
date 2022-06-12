# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 22:15:12 2022
Compare Elm Index vs. SPY and VTI
@author: vragu
"""

import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
#from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick

# Load Elm Index
elm = pd.read_pickle('../data/sim_port_rets.pickle')
sym = 'SPY'

# Load data for SPY and VTI
start =  elm.index[0]
end   =  elm.index[-1]

ticker = web.DataReader(sym, 'yahoo', start, end)

# Append Elm to SPY and VTI series
df = pd.merge_asof(ticker, elm, left_index=True, right_index=True)

# Drop rows where we don't have ticker returns 
df1 = (df[df['simple_rets'] != 0]).copy()

# Drop columns that we don't need
df1.drop(['High','Low','Open','Close','Volume', 'simple_rets'],axis=1, inplace=True)

# Create return series
df1[sym] = df1['Adj Close']/ df1['Adj Close'][0]
df1['Elm']  = df1['cum_rets']/ df1['cum_rets'][0]

dfm = df1.resample('M').last()[[sym, 'Elm']]
dfm[f'Elm/{sym}'] = dfm['Elm'] / dfm[sym]

#%% Plot Cumulative Returns with labels
years = (dfm.index[-1] - df.index[0]) / timedelta(days=365.25) + 1/12
elm_avg_ret = dfm['Elm'][-1] ** (1/years) - 1
idx_avg_ret = dfm[sym][-1]   ** (1/years) - 1
diff_ret = elm_avg_ret - idx_avg_ret

ax = dfm.plot(logy=True, title=f"Historical Cumulative Returns: Elm vs. {sym}\n(Ticker : Average Annual Return)")
ax.legend([f'Elm : {elm_avg_ret*100:.2f}%',
           f'{sym} : {idx_avg_ret*100:.2f}%',
           f'Elm/{sym} : {diff_ret*100:.2f}%'], fontsize='small')
ax.set_xlabel(None)
plt.show()

#%% Plot Rolling 1y Returns
avg_year = 3
dfret = np.log(dfm).diff(avg_year * 12)/avg_year
ax = (dfret[f'Elm/{sym}']*100).plot(title=f"Historical {avg_year}-year Rolling Return Difference: Elm vs. {sym}")
ax = ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
ax.set_xlabel(None)
plt.show()

print("Done")