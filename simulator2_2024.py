# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:50:29 2024

Simulate a strategy of going long/short portfolio of top/bottom deciles based on OBP
Unlike the first simulator, which uses market caps and calculates its own weights, 
this version loads index weights from a file
@author: vragu
"""

# %% Load modules
# import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import load_FF_rates as ff
from datetime import timedelta
import datetime as dt

# import os

# %% Global variables - file locations
TICKER_FILE = "../data/elm_good_tkrs.pickle"
ALL_STOCKS_DF = "../data/elm_stocks_px_mcap.pickle"
STOCK_DFS_DIR = "../stock_dfs"
MONTHLY_FIELD_DF = "../data/elm_monthly_{field}.pickle"
IDX_WEIGHT_FILE = "../data/cap1_idx_weights.pickle"

TD_PER_MONTH = 21.6  # numbe of trading days per month


# %% Initialize Simulation Parameters
def init_sim_params():
	""" Initialize Sim Params, return dictionary"""
	params = {}
	params['rebuild'] = False  # Regenerate stock return files
	params['window'] = 24  # window in months for the return calculation
	params['trx_costs'] = 0.5  # one-way trading costs in bp
	params['borrow_fee'] = 25  # borrow fee on the shorts, in bp per annum
	params['capital'] = 1  # initial capital
	params['trade_pctiles'] = [20, 80]  # Sell and buy  thresholds for shorts/longs portfolios
	params['gross'] = True  # Whether to produce graphs and reports for gross or net returns
	params['max_weight'] = None  # Max weight in a portfolio as a multiple of equal-weights
	params['min_weight'] = None  # Min weight
	params[
		'short_gap'] = None  # If we exclude days with a long gap: True - Exclude, False - Only long gaps, None-ignore

	# Load index weights for the stock universe (we assume that our positions are proportional to these weights)
	# These can be thought of as market caps (which are re-scaled that total weight for both long and short portfoio
	# adds up to 1)
	params['idx_weights'] = pd.read_pickle(IDX_WEIGHT_FILE).astype(float)
	return params


# %% Load all stock panel data from pickle files
def load_data():
	with open(TICKER_FILE, "rb") as f:
		tickers = pickle.load(f)

	with open(ALL_STOCKS_DF, "rb") as f:
		df_all_stocks = pickle.load(f)

	return tickers, df_all_stocks


# %% Build a monthly dataframe of overnight returns
def load_monthly_df(field='r_ovnt', rebuild=False, save_df=True, short_gap=None):
	""" Parameters:
            field   - name of a field that we want to load for every stock
            rebuild - if True, generate monthly o/n return from panel data
                     else, load from a pickle file
            save_df - save the resulting dataframe into a pickle
            short_gap - if True - only 
            include days for which t-1 was also a trading day
                        if False - only include days for which t-1 was not a trading day
                        if None - include both
    """

	pickle_fname = MONTHLY_FIELD_DF.format(field=field)

	if not rebuild:
		# Load file from pickle
		dfm = pd.read_pickle(pickle_fname)
	else:

		# Re-generate monthy on returns form panel data
		tickers, df_all_stocks = load_data()
		# df1 = df_all_stocks.pivot_table(values='Value', index='Date', columns=['Ticker', 'Field'])
		df1 = df_all_stocks.pivot(values='Value', index='Date', columns=['Ticker', 'Field'])

		# Extract Intraday Returns for all stocks
		df2 = df1.swaplevel(axis=1)[field].astype(float)

		# Mask out returns with long gaps
		if short_gap is not None:
			df2['_date_'] = df2.index
			df2['gap'] = df2['_date_'].diff() / dt.timedelta(days=1)
			if short_gap == True:  # Only consider days with short gap
				df2[df2['gap'] > 1] = 0
			else:
				df2[df2['gap'] == 1] = 0
			df2.drop(['_date_', 'gap'], axis=1, inplace=True)  # Drop columns that we don't need anymore

		# Resample to monthly frequency
		dfm = df2.resample('M').sum()

		# Save new df into a pickle
		if save_df:
			dfm.to_pickle(pickle_fname)

	return dfm


# %% Generate positions for the simulation based on a sort indicator
def gen_positions(df_sort, sim_params):
	""" Generate stock positions
        Params:
            df_sort - dataframe with sort indicator for each date an stock
            sim_params - dictionary with sort parameter
       Return: 
           positions  - dataframe with positions (+1/0/-1) for all stocks and dates
           thresholds - dataframe with values of thresholds used to form percentiles
    """

	# Unpack parameters
	window = sim_params['window']  # window over which we calc obp

	# Generate top and bottom quantiles on the basis of the sorting parameter
	trade_pctiles = sim_params['trade_pctiles']  # Sell and buy  thresholds for shorts/longs portfolios

	# Generate positions
	# OBP thresholds long/short portfolios
	thresholds = pd.DataFrame(np.nan, index=df_sort.index, columns=['buy', 'sell', 'n_buys', 'n_sells'])

	# Positions in each stock at the close of a given date close
	positions = pd.DataFrame(np.nan, index=df_sort.index, columns=df_sort.columns)

	for t, date in enumerate(df_sort.index):
		if t < window - 1:
			continue

		row = df_sort.loc[date, :].values
		non_zero_rets = row[row != 0]

		# Calculate thresholds for long/short portfolio
		t_sell, t_buy = np.percentile(non_zero_rets, trade_pctiles)

		# Calculate positions for all stocks
		positions.loc[date, :] = np.where(df_sort.loc[date, :] >= t_buy, 1, 0)
		positions.loc[date, :] = np.where(df_sort.loc[date, :] <= t_sell, -1, positions.loc[date, :])

		# Calculate the number of longs and shorts
		n_buys = (positions.loc[date, :] == 1).values.sum()
		n_sells = (positions.loc[date, :] == -1).values.sum()

		# print(f"t={t}, t_buy={t_buy:.3f}, t_sell={t_sell:.3f}, n_buys={n_buys}, n_sells={n_sells}")

		thresholds.loc[date, :] = t_buy, t_sell, n_buys, n_sells

	return positions, thresholds


# %% Generate portfoio weights that add up to 1 for a subset of tickers
def gen_port_weights(date, mask, params):
	"""Calculate portfolio weights that add up to 1 for a subset of stocks given market caps
    Parameters:
        date:   datetime type, should be in the rows index
        mask:   boolean array to indicate stocks which we should include
        params: dict with parameters
        
    Return: 
        weights: pd series with weights for every stock
    """

	# If there are stocks that we don't want to trade, exclude them
	# do_not_trade = params['do_not_trade']
	# if do_not_trade is not None:
	#     for ticker in do_not_trade:
	#         mask[ticker] = False

	hist_weights = params['idx_weights']
	weights_on_date = hist_weights.loc[date, :]
	raw_weights = (weights_on_date * mask) / np.sum(weights_on_date * mask)

	# Check that all weights are within allowed range
	if (params['max_weight'] is None) and (params['min_weight'] is None):
		weights = raw_weights
	else:
		if params['max_weight'] is not None:
			max_weight = params['max_weight'] / sum(mask)
			raw_weights[raw_weights > max_weight] = max_weight

		if params['min_weight'] is not None:
			min_weight = params['min_weight'] / sum(mask)
			raw_weights[(raw_weights < min_weight) & (raw_weights > 0)] = min_weight

		weights = raw_weights / raw_weights.sum()

	print(f"Max={weights[mask].max() * 100:.2f}, min={weights[mask].min() * 100:.2f}")
	return weights


# %% Calculate portfolio returns
def calc_portfolio_returns(pos, ret, ret_full, riskfree, params, intra=False):
	""" Calculate portfolio returns
        Parameters:
            pos      - dataframe of positions for each period
            ret      - dataframe of partial-day returns (o/n or intraday) for each period
            ret_full - dataframe of full day returns
            riskfree - risk-free rate returns (same index as other dataframes)
            params   - dictionary with parameters
            intra    - boolean, True is the strategy does not hold positions overnight
            
        Return: pandas dataframe  of simple returns
    """

	# Unpack parameters
	try:
		window = params['window']
		borrow_fee = params['borrow_fee']
		trx_costs = params['trx_costs']

	except KeyError as err:
		print("Missing simulation parameter: ", err)

	# Set up a dataframe to hold the results
	port_cols = ['r_l', 'r_s', 'r_ls', 'r_ix', 'r_ix_f']
	port_r = pd.DataFrame(np.nan, index=ret.index, columns=port_cols)
	port_r_net = pd.DataFrame(np.nan, index=ret.index, columns=port_cols)

	# Convert exponential to simlpe returns, so we can aggregate the cross-section
	ret_s = np.exp(ret) - 1
	ret_full_s = np.exp(ret_full) - 1

	# Populate dataframe with portfolio returns
	for t, date in enumerate(ret.index):
		if t <= window - 1:
			continue

		date_prev = ret.index[t - 1]

		# Extract position and return rows
		pos_row = pos.iloc[t - 1, :]
		ret_row = ret_s.iloc[t, :]
		ret_full_row = ret_full_s.iloc[t, :]

		# Identify long and short stocks, and stocks with missing data
		long_mask = (pos_row == 1)
		shrt_mask = (pos_row == -1)
		zero_mask = (ret_row == 0) & (ret_full_row == 0)

		# Generate weights of long, short and index portfolios, shift by 1 so that the align with returns
		long_weights = gen_port_weights(date_prev, long_mask, params)
		shrt_weights = -gen_port_weights(date_prev, shrt_mask, params)
		indx_weights = gen_port_weights(date_prev, ~zero_mask, params)

		# =============================================================================
		# Gross returns on longs, shorts, long/short, index o/n or intra, index buy and hold
		# =============================================================================
		stock_r_long = long_weights * ret_row
		stock_r_shrt = shrt_weights * ret_row
		stock_r_indx = indx_weights * ret_row
		stock_r_indx_full = indx_weights * ret_full_row

		if not intra:  # if we hold positions overnight
			port_r.loc[date, 'r_l'] = stock_r_long.sum()
			port_r.loc[date, 'r_s'] = stock_r_shrt.sum() + riskfree.loc[date, 'Rets'] * 2
			port_r.loc[date, 'r_ix'] = stock_r_indx.sum()

		else:  # no positions overnight - pay no interest or borrow fees
			port_r.loc[date, 'r_l'] = stock_r_long.sum() + riskfree.loc[date, 'Rets']
			port_r.loc[date, 'r_s'] = stock_r_shrt.sum() + riskfree.loc[date, 'Rets']
			port_r.loc[date, 'r_ix'] = stock_r_indx.sum() + riskfree.loc[date, 'Rets']

		port_r.loc[date, 'r_ls'] = port_r.loc[date, 'r_l'] + port_r.loc[date, 'r_s']

		# Index buy-and-hold return - subtract initial transactions costs, otherwise assume no trading
		# Only consider stocks for which we have returns (i.e. ones that trade)
		port_r.loc[date, 'r_ix_f'] = stock_r_indx_full.sum()

		# =============================================================================
		# Net returns - assume we trade 2x per day on each position, TD_PER_MONTH trading days per month x 2 trades
		# =============================================================================
		monthly_trx_costs = trx_costs * 2 * TD_PER_MONTH / 10000

		if not intra:
			port_r_net.loc[date, 'r_l'] = port_r.loc[date, 'r_l'] - monthly_trx_costs
			port_r_net.loc[date, 'r_s'] = port_r.loc[date, 'r_s'] - monthly_trx_costs - borrow_fee / 12 / 10000
			port_r_net.loc[date, 'r_ix'] = port_r.loc[date, 'r_ix'] - monthly_trx_costs

		else:  # no positions overnight - pay no interest or borrow fees
			port_r_net.loc[date, 'r_l'] = port_r.loc[date, 'r_l'] - monthly_trx_costs
			port_r_net.loc[date, 'r_s'] = port_r.loc[date, 'r_s'] - monthly_trx_costs
			port_r_net.loc[date, 'r_ix'] = port_r.loc[date, 'r_ix'] - monthly_trx_costs

		port_r_net.loc[date, 'r_ls'] = port_r_net.loc[date, 'r_l'] + port_r_net.loc[date, 'r_s']

		# Index buy-and-hold return
		if t == window:
			port_r_net.loc[date, 'r_ix_f'] = port_r.loc[date, 'r_ix_f'] - trx_costs / 10000
		else:
			port_r_net.loc[date, 'r_ix_f'] = port_r.loc[date, 'r_ix_f']

	return port_r, port_r_net


# %% Generate cumulative return plots
def plot_sim_returns(ret, use_log=True, start=None, end=None, title_codes=None, logy=True):
	# Extract returns over the focus period
	ret1 = ret[start:end].dropna()

	# Cumulative returns
	ret_cum = (1 + ret1).cumprod()
	rln_cum = np.log(ret_cum)

	if use_log:
		title_log_str = "(Log) "
	else:
		title_log_str = ""

	# Set up labels for the plot
	legend_template = ['Long {}', 'Short {}', 'L/S {}', 'Index {}', 'Index BuyHold']
	linestyles = ['-', '-', '-', ':', ':']

	if sim['trade_pctiles'][0] == 20:
		pctile_str = "Top/Bottom Quintiles"
	elif sim['trade_pctiles'][0] == 10:
		pctile_str = "Top/Bottom Deciles"
	else:
		pctile_str = "{}/{} Quantiles".format(*sim['trade_pctiles'])

	title_string = "Equity Curves {} - {} - {}\n{} on Trailing (O/N-Intraday), {}" \
		.format(title_log_str, title_codes[2], title_codes[0], \
	            pctile_str, title_codes[1])

	if use_log:
		ax = rln_cum.plot(title=title_string, style=linestyles, fontsize='small')
	else:
		capital = sim['capital']
		ax = (ret_cum * capital).plot(title=title_string, style=linestyles, fontsize='small', logy=logy)

	if title_codes[0] == 'Overnight':
		period_label = 'O/N'
	elif title_codes[0] == 'Intraday':
		period_label = 'Intra'
	else:
		period_label = 'Full'

	legend_list = [x.format(period_label) for x in legend_template]
	ax.legend(legend_list, fontsize='small')
	ax.set_ylabel("Capital $", fontsize='small')
	ax.set_xlabel("Date", fontsize='small')
	plt.show()

	# Return cumulative log returns
	return rln_cum


# %% Generate cumulative return plots
def plot_for_paper(ret, use_log=True, start=None, end=None, title_codes=None):
	# Extract returns over the focus period
	ret1 = ret[start:end].dropna()

	# Cumulative returns
	ret_cum = (1 + ret1).cumprod()
	rln_cum = np.log(ret_cum)

	if use_log:
		title_log_str = "(Log) "
	else:
		title_log_str = ""

	# Set up labels for the plot
	legend_template = ['Long {}', 'Short {}', 'L/S {}', 'Index {}', 'Index BuyHold']
	linestyles = ['-', '-', '-', ':', ':']

	if sim['trade_pctiles'][0] == 20:
		pctile_str = "Top/Bottom Quintiles"
	elif sim['trade_pctiles'][0] == 10:
		pctile_str = "Top/Bottom Deciles"
	else:
		pctile_str = "{}/{} Quantiles".format(*sim['trade_pctiles'])

	title_string = "Value of $1 Invested {} - {} - {}\n{} on Trailing {}y [O/N-Intraday], {}" \
		.format(title_log_str, title_codes[2], title_codes[0], \
	            pctile_str, sim['window'] / 12, title_codes[1])

	if use_log:
		ax = rln_cum.plot(title=title_string, style=linestyles, fontsize='small')
	else:
		capital = sim['capital']
		ax = (ret_cum * capital).plot(title=title_string, style=linestyles, fontsize='small', logy=True)

	if title_codes[0] == 'Overnight':
		period_label = 'O/N'
	else:
		period_label = 'Intra'

	legend_list = [x.format(period_label) for x in legend_template]
	ax.legend(legend_list, fontsize='small')
	ax.set_ylabel("Capital $", fontsize='small')
	# ax.set_xlabel("Date", fontsize='small')
	ax.set_xlabel(None)
	ax.yaxis.set_major_formatter(FormatStrFormatter('$%.1f'))
	# for label in ax.get_xticklabels(which='major'):
	#    label.set(rotation=30)
	plt.show()

	# Return cumulative log returns
	return rln_cum


# %% Generate a simple plot for paper - only L/S but nicely labelled
def plot_simple_LS(port_grs, port_net, start=None, end=None, code='on'):
	""" Plot a simplified graph, only LS gross and net"""

	assert code == 'on', 'Not implemented for other holding periods'

	ret_grs = port_grs['r_ls'].dropna()
	ret_net = port_net['r_ls'].dropna()

	df = pd.DataFrame({'r_grs': ret_grs, 'r_net': ret_net})

	df['cum_grs'] = (1 + df['r_grs']).cumprod()
	df['cum_net'] = (1 + df['r_net']).cumprod()
	df['5y_avg_grs'] = df['r_grs'].rolling(60).mean() * 12
	df['5y_avg_net'] = df['r_net'].rolling(60).mean() * 12

	# Build up the graph of cumulative returns
	title_string = ("Value of $1 Invested - 1995-2024- Holding Overnight"
	                f"\nLong/Short Portfolios based on Trailing {sim['window']}y [O/N-Intraday] Quintiles")
	linestyles = [':', '-']

	ax = df[['cum_grs', 'cum_net']].plot(logy=True, style=linestyles, fontsize='small', grid=True)
	ax.set_title(title_string, fontsize=10)

	legend_list = ["Gross (Final=${:.0f})".format(df['cum_grs'].iloc[-1]),
	               "Net   (Final=${:.0f})".format(df['cum_net'].iloc[-1])]

	ax.legend(legend_list, fontsize='small')
	ax.set_ylabel("Capital $", fontsize='small')
	ax.set_xlabel(None)
	ax.yaxis.set_major_formatter(FormatStrFormatter('$%.1f'))

	box_text = ("Net Return: 30.8%\n"
	            "Standard Deviation: 9.5%\n"
	            "Sharpe Ratio Net: 2.65\n"
	            "Largest Drawdown: 5.3%\n"
	            "t-Stat: 13.9")
	ax.text(dt.datetime(2019, 1, 1), 1, box_text, fontsize='small',
	        bbox=dict(facecolor='white', edgecolor='none'), horizontalalignment='right')

	plt.show()

	# Graph of rolling returns
	years = (df.index[-1] - df.index[0]) / timedelta(days=365.25)
	grs_avg_ret = df['cum_grs'][-1] ** (1 / years) - 1
	net_avg_ret = df['cum_net'][-1] ** (1 / years) - 1

	title_string = ("Rolling 5y Annual Returns - 1995-2024- Holding Overnight"
	                f"\nLong/Short Portfolios based on Trailing {sim['window']}y [O/N-Intraday] Quintiles")
	linestyles = [':', '-']
	ax = (df[['5y_avg_grs', '5y_avg_net']] * 100).plot(style=linestyles, fontsize='small', grid=True)
	ax.set_title(title_string, fontsize=10)

	legend_list = ["Gross (Avg={:.1f}%)".format(grs_avg_ret * 100),
	               "Net   (Avg={:.1f}%)".format(net_avg_ret * 100)]

	ax.legend(legend_list, fontsize='small')
	ax.set_ylabel("Rolling 5y Annual return %", fontsize='small')
	ax.set_xlabel(None)
	ax = ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

	plt.show()

	return df


# %% Generate a simple gross only plot for paper - only L/S but nicely labelled
def plot_simple_LS_gross(port_grs, start=None, end=None, code='on'):
	""" Plot a simplified graph, only LS gross only"""

	ret_grs = port_grs['r_ls'].dropna()

	df = pd.DataFrame({'r_grs': ret_grs})

	df['cum_grs'] = (1 + df['r_grs']).cumprod()
	df['5y_avg_grs'] = df['r_grs'].rolling(60).mean() * 12

	# Build up the graph of cumulative returns
	if code == 'on':
		holding_code = "Overnight"
		logy = True
		box_text = ("Gross Return (L/S):  38.1%\n"
		            "Long:  +27.2% / Short:  +7.9%\n"
		            "Standard Deviation (L/S):  10.9%\n"
		            "Sharpe Ratio Gross (L/S):  2.8\n"
		            "Largest Drawdown:  10.9%\n"
		            "t-Stat:  14.8")
		legend_loc = 'best'
	elif code == 'intra':
		holding_code = "Intraday"
		logy = True
		# box_text = ("Gross Return (L/S):  38.1%\n"
		#             "Long:  +27.2% / Short:  +7.9%\n"
		#             "Standard Deviation (L/S):  10.9%\n"
		#             "Sharpe Ratio Gross (L/S):  2.8\n"
		#             "Largest Drawdown:  10.9%\n"
		#             "t-Stat:  14.8")
		legend_loc = 'best'
	else:
		holding_code = "Full Day"
		logy = False
		box_text = ("Gross Return (L/S):  6.1%\n"
		            "Long:  +12.0% / Short:  -7.5%\n"
		            "Standard Deviation (L/S):  17.3%\n"
		            "Sharpe Ratio Gross (L/S):  0.3\n"
		            "Largest Drawdown:  58.5%\n"
		            "t-Stat:  1.5")
		legend_loc = 'upper left'

	# linestyles  = [':','-']

	ax = df['cum_grs'].plot(logy=logy, fontsize='small', grid=True)

	title_string = (f"Value of $1 Invested - 1995-2024- Holding {holding_code}"
	                f"\nLong/Short Portfolios based on Trailing {sim['window']}m [O/N-Intraday] Quintiles")

	ax.set_title(title_string, fontsize=10)

	legend_list = ["Gross Return (Final=${:.0f})".format(df['cum_grs'].iloc[-1])]

	ax.legend(legend_list, loc=legend_loc, fontsize='small', fancybox=True, framealpha=0.5)
	ax.set_ylabel("Capital $", fontsize='small')
	ax.set_xlabel(None)
	ax.yaxis.set_major_formatter(FormatStrFormatter('$%.1f'))

	add_text = False
	if add_text:
		ax.text(dt.datetime(2019, 1, 1), 1, box_text, fontsize='small',
		        bbox=dict(facecolor='white', edgecolor='none'), horizontalalignment='right')

	plt.show()

	# Graph of rolling returns
	years = (df.index[-1] - df.index[0]) / timedelta(days=365.25)
	grs_avg_ret = df['cum_grs'][-1] ** (1 / years) - 1

	title_string = (f"Rolling 5y Annual Returns - 1995-2024- Holding {holding_code}"
	                f"\nLong/Short Portfolios based on Trailing {sim['window']}m [O/N-Intraday] Quintiles")

	# linestyles  = [':','-']
	ax = (df['5y_avg_grs'] * 100).plot(fontsize='small', grid=True)
	ax.set_title(title_string, fontsize=10)

	legend_list = ["Gross (Avg={:.1f}%)".format(grs_avg_ret * 100)]

	ax.legend(legend_list, fontsize='small')
	ax.set_ylabel("Rolling 5y Annual return %", fontsize='small')
	ax.set_xlabel(None)
	ax = ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

	plt.show()

	return df


# %% Generate time series statistics
def get_time_series_stats(rets, riskfree, annualize=True):
	""" Generate descriptive statistics for a time series
        Params: rets     - pandas series exponential period returns
                riskfree - pandas series of exponential riskfree returns
                
        Output: tuple: Mean Ret, Geom Ret, Vol, Sharpe - all annualized
    """

	# Calculate a series of excess return
	xret = rets - riskfree

	# Simple mean return
	xret_s = np.exp(xret) - 1
	xr_mean = xret_s.mean()

	# Geometric return (aka CAGR)
	xr_geom = np.exp(xret.mean()) - 1

	# Volatility
	xr_vol = xret.std()

	# If required convert from monthly to annual
	if annualize:
		xr_mean *= 12
		xr_geom *= 12
		xr_vol *= np.sqrt(12)

	xr_sharpe = xr_mean / xr_vol

	return xr_mean, xr_geom, xr_vol, xr_sharpe


# %% Generate summary report
def gen_summary_report():
	""" Don't pass any parameters, get all info from the namespace
        Assume that the global namespace contains frames referenced below with cumulative strategy returns
    """

	# Set up a dataframe to hold the results
	df_stats = pd.DataFrame(np.nan, index=range(18), \
	                        columns=['Period', 'Window', 'Series', 'Costs', 'Mean Ret', 'Geom Ret', 'Vol', 'Sharpe'])

	# Set up iterables
	frames = [r_o_pre, r_i_pre, r_o_post, r_i_post]
	periods = ['Pre-2015', 'Pre-2015', 'Post-2015', 'Post-2015']
	windows = ['O/N', 'Intra', 'O/N', 'Intra']

	series_names = ['Long', 'Short', 'L/S', 'Index']
	col_names = ['r_l', 'r_s', 'r_ls', 'r_ix']
	stats_names = df_stats.columns[4:]

	i = 0
	for frame, period, window in zip(frames, periods, windows):
		# Add 'active' strategy returns
		for s_name, s_col in zip(series_names, col_names):
			# Pick a return series out of an input simulation dataframe
			s = frame[s_col]

			# Calculate stats for the return series (need to diff since s is cumulative)
			stats = get_time_series_stats(s.diff(), riskfree['Rets'])

			# Populat a row  in the summary dataframe
			df_stats.loc[i, ['Period', 'Window', 'Series']] = [period, window, s_name]
			df_stats.loc[i, stats_names] = stats

			# Increment row counter
			i += 1

	# Add rows for the index buy and hold returns
	for j in [0, 2]:
		frame, period = frames[j], periods[j]

		# Pick the series with the buy-and-hold index return
		s_name = 'Index'
		s_col = 'r_ix_f'
		window = 'BuyHold'

		# Pick a return series out of an input simulation dataframe
		s = frame[s_col]

		# Calculate stats for the return series (need to diff since s is cumulative)
		stats = get_time_series_stats(s.diff(), riskfree['Rets'])

		# Populat a row  in the summary dataframe
		df_stats.loc[i, ['Period', 'Window', 'Series']] = [period, window, s_name]
		df_stats.loc[i, stats_names] = stats

		# Increment row counter
		i += 1

	# Populate the Costs (Gross/Net) column
	df_stats['Costs'] = "Gross" if sim['gross'] else "Net"

	# Print results
	print(df_stats)

	return df_stats


# %% Generate a bespoke table for the paper (based on gen_summary_report, but with modifications)
def gen_table_for_paper():
	""" Don't pass any parameters, get all info from the namespace
        Assume that the global namespace contains frames referenced below with cumulative strategy returns
    """

	# Set up a dataframe to hold the results
	df_stats = pd.DataFrame(np.nan, index=range(18), \
	                        columns=['Period', 'Window', 'Series', 'Costs', 'Mean Ret', 'Geom Ret', 'Vol', 'Sharpe'])

	# Set up iterables
	frames = [r_o_pre, r_i_pre, r_o_post, r_i_post, r_o_full, r_i_full]
	periods = ['Pre-2015', 'Pre-2015', 'Post-2015', 'Post-2015', 'Full', 'Full']
	windows = ['O/N', 'Intra', 'O/N', 'Intra', 'O/N', 'Intra']

	series_names = ['Long', 'Short', 'L/S', 'Index', 'Idx_BuyHold']
	col_names = ['r_l', 'r_s', 'r_ls', 'r_ix', 'r_ix_f']
	stats_names = df_stats.columns[4:]

	i = 0
	for frame, period, window in zip(frames, periods, windows):
		# Add 'active' strategy returns
		for s_name, s_col in zip(series_names, col_names):
			# Pick a return series out of an input simulation dataframe
			s = frame[s_col]

			# Calculate stats for the return series (need to diff since s is cumulative)
			stats = get_time_series_stats(s.diff(), riskfree['Rets'])

			# Populat a row  in the summary dataframe
			df_stats.loc[i, ['Period', 'Window', 'Series']] = [period, window, s_name]
			df_stats.loc[i, stats_names] = stats

			# Increment row counter
			i += 1

	# Add rows for the index buy and hold returns
	for j in [0, 2, 4]:
		frame, period = frames[j], periods[j]

		# Pick the series with the buy-and-hold index return
		s_name = 'Index'
		s_col = 'r_ix_f'
		window = 'BuyHold'

		# Pick a return series out of an input simulation dataframe
		s = frame[s_col]

		# Calculate stats for the return series (need to diff since s is cumulative)
		stats = get_time_series_stats(s.diff(), riskfree['Rets'])

		# Populate a row  in the summary dataframe
		df_stats.loc[i, ['Period', 'Window', 'Series']] = [period, window, s_name]
		df_stats.loc[i, stats_names] = stats

		# Increment row counter
		i += 1

	# Populate the Costs (Gross/Net) column
	df_stats['Costs'] = "Gross" if sim['gross'] else "Net"

	# Print results
	print(df_stats)

	return df_stats


# %% Generate a simple table for the paper
def gen_simple_table(port_grs, port_net, use_net=True):
	""" Generate a simlpe table for the paper with gross/net returns and related stats"""

	cost_code = "Net" if use_net else 'Gross'

	# Column title for the table
	col_names = {'r_ls': 'Long vs. Short',
	             'r_l': 'Long Side',
	             'r_s': 'Short Side'}

	row_titles = ["Value of $1 Invested Before Trx Costs",
	              "Value of $1 Invested After Trx Costs",
	              f"{cost_code} Return pa",
	              "Standard Deviation pa",
	              f"Sharpe Ratio {cost_code}",
	              "Largest Drawdown",
	              "t-Stat"]

	dfs = pd.DataFrame(columns=col_names.values(), index=row_titles)

	# =============================================================================
	#  Define a calculation for each column
	# =============================================================================
	def gen_one_col_simple_table(s_label, use_net=use_net):
		""" Generate one column of the simple table"""

		cost_code_shrt = "net" if use_net else "grs"

		col = pd.Series(0.0, index=row_titles)

		# Get gross and net series and calculate cumulative
		ret_grs = port_grs[s_label]
		ret_net = port_net[s_label]

		df = pd.DataFrame({'r_grs': ret_grs, 'r_net': ret_net}).dropna()

		df['cum_grs'] = (1 + df['r_grs']).cumprod()
		df['cum_net'] = (1 + df['r_net']).cumprod()

		# Populate the columns
		# Mean return
		col["Value of $1 Invested Before Trx Costs"] = df['cum_grs'].iloc[-1]
		col["Value of $1 Invested After Trx Costs"] = df['cum_net'].iloc[-1]

		# Annual return
		years = (df.index[-1] - df.index[0]) / timedelta(days=365.25)
		col[f"{cost_code} Return pa"] = df[f'cum_{cost_code_shrt}'][-1] ** (1 / years) - 1

		# Std Dev
		col["Standard Deviation pa"] = np.log(1 + df[f'r_{cost_code_shrt}']).std() * np.sqrt(12)

		# Sharpe Ratio
		df1 = df.merge(riskfree, how='left', left_index=True, right_index=True)
		df1[f'xr_{cost_code_shrt}'] = df1[f'r_{cost_code_shrt}'] - df1['Rets']  # second term is risk-free rate

		col[f"Sharpe Ratio {cost_code}"] = df1[f'xr_{cost_code_shrt}'].mean() * 12 / col["Standard Deviation pa"]

		# Largest drawdown
		df1['drawdown'] = 1 - df[f'cum_{cost_code_shrt}'] / df[f'cum_{cost_code_shrt}'].cummax()
		col["Largest Drawdown"] = df1['drawdown'].max()

		# T-stat
		col['t-Stat'] = col[f"Sharpe Ratio {cost_code}"] * np.sqrt(years)

		return col

	# =============================================================================
	#  Loop over time series and generate a column for each
	# =============================================================================
	for s, col in col_names.items():
		dfs[col] = gen_one_col_simple_table(s, use_net=use_net)

	return dfs


# %% Start of the main program
if __name__ == "__main__":

	# Initialize simulation parameters
	sim = init_sim_params()

	# Generate dataframes of monthly overnight and intraday returns
	df_o = load_monthly_df(field='r_ovnt', rebuild=sim['rebuild'], save_df=True, short_gap=sim['short_gap'])
	df_i = load_monthly_df(field='r_intr', rebuild=sim['rebuild'], save_df=True, short_gap=sim['short_gap'])
	df_f = df_o + df_i  # full day return

	# Load Fed Funds returns (the risk-free rate)
	riskfree = ff.load_FF_period_rets(reload=sim['rebuild'], start=df_o.index[0], end=df_o.index[-1])

	# Calculate the sorting parameter, for now just use (o/n - intra) window, later can do opb as in Lachance
	df_obp = (df_o - df_i).rolling(sim['window']).sum()

	# Generate positions and thresholds used for long and short portfolio cutoffs
	positions, _ = gen_positions(df_obp, sim)
	positions.to_csv('../data/positions2.csv')
	positions.to_pickle('../data/positions2.pickle')

	# Calculate portfolio monthly and cumulative returns, with and w/o trans costs
	port_o, port_o_net = calc_portfolio_returns(positions, df_o, df_f, riskfree, sim)
	port_i, port_i_net = calc_portfolio_returns(-positions, df_i, df_f, riskfree, sim, intra=True)
	port_f, port_f_net = (1 + port_o) * (1 + port_i) - 1, (1 + port_o_net) * (1 + port_i_net) - 1

	# %% Plot returns for the overnight holding strategy
	use_log = False
	costs = "Gross" if sim['gross'] else "Net"

	port_o_used = port_o_net if costs == "Net" else port_o
	port_i_used = port_i_net if costs == "Net" else port_i
	port_f_used = port_f_net if costs == "Net" else port_f

	# Before 2015
	r_o_pre = plot_sim_returns(port_o_used, end='2015-01-01', title_codes=['Overnight', costs, '1995-2015'],
	                           use_log=use_log)
	r_i_pre = plot_sim_returns(port_i_used, end='2015-01-01', title_codes=['Intraday', costs, '1995-2015'],
	                           use_log=use_log)
	r_f_pre = plot_sim_returns(port_f_used, end='2015-01-01', title_codes=['Full day', costs, '1995-2015'],
	                           use_log=use_log)

	# After 2015
	r_o_post = plot_sim_returns(port_o_used, start='2015-01-01', title_codes=['Overnight', costs, '2015-2024'],
	                            use_log=use_log)
	r_i_post = plot_sim_returns(port_i_used, start='2015-01-01', title_codes=['Intraday', costs, '2015-2024'],
	                            use_log=use_log)
	r_f_post = plot_sim_returns(port_f_used, start='2015-01-01', title_codes=['Full day', costs, '2015-2024'],
	                            use_log=use_log)

	# Last 2 years
	# r_o_post22 = plot_sim_returns(port_o_used, start='2022-05-01', title_codes=['Overnight', costs, '2022-2024'],
	#                               use_log=use_log, logy=False)
	# r_i_post22 = plot_sim_returns(port_i_used, start='2022-05-01', title_codes=['Intraday', costs, '2022-2024'],
	#                               use_log=use_log, logy=False)
	# r_f_post22 = plot_sim_returns(port_f_used, start='2022-05-01', title_codes=['Full day', costs, '2022-2024'],
	#                               use_log=use_log, logy=False)

	# This year
	r_o_24 = plot_sim_returns(port_o_used, start='2023-12-31', title_codes=['Overnight', costs, '12/31/23-5/15/24'],
	                          use_log=use_log, logy=False)
	r_i_24 = plot_sim_returns(port_i_used, start='2023-12-31', title_codes=['Intraday', costs, '12/31/23-5/15/24'],
	                          use_log=use_log, logy=False)
	r_f_22 = plot_sim_returns(port_f_used, start='2023-12-31', title_codes=['Full day', costs, '12/31/23-5/15/24'],
	                          use_log=use_log, logy=False)

	# Customizer plot for the paper
	report_for_paper = True
	if report_for_paper:
		r_o_full = plot_for_paper(port_o_used, title_codes=['Overnight', costs, '1995-2024'], use_log=use_log)
		r_i_full = plot_for_paper(port_i_used, title_codes=['Intraday', costs, '1995-2024'], use_log=use_log)
		r_f_full = plot_for_paper(port_f_used, title_codes=['Full day', costs, '1995-2024'], use_log=use_log)

	# %% Analyze the cumulative return time series
	if report_for_paper:
		print("\nFull table for the paper:")
		df_stats = gen_table_for_paper()
		fname = "df_stats_paper.p"
	else:
		print("\nSummary report:")
		df_stats = gen_summary_report()
		fname = "df_paper.p"

	# %% Generate a simple table for the paper
	use_net = False
	print("\nSimple table for the paper -- overnight:")
	if use_net:
		df_o_simple_plot = plot_simple_LS(port_o, port_o_net, code='on')
		df_o_simple = gen_simple_table(port_o, port_o_net)
	else:
		df_o_simple_plot = plot_simple_LS_gross(port_o, port_o_net, code='on')
		df_o_simple = gen_simple_table(port_o, port_o_net, use_net=False)
	print(df_o_simple)

	print("\nSimple table for the paper -- intraday:")
	if use_net:
		df_i_simple_plot = plot_simple_LS(port_i, port_i_net, code='intra')
		df_i_simple = gen_simple_table(port_i, port_i_net)
	else:
		df_i_simple_plot = plot_simple_LS_gross(port_i, port_i_net, code='intra')
		df_i_simple = gen_simple_table(port_i, port_i_net, use_net=False)
	print(df_i_simple)

	# %% Now run analysis for the full day:
	print("\nSimple table for the paper -- Full day:")
	if use_net:
		df_f_simple_plot = plot_simple_LS(port_f, port_f_net, code='full')
		df_f_simple = gen_simple_table(port_f, port_f_net)
	else:
		df_f_simple_plot = plot_simple_LS_gross(port_f, port_f_net, code='full')
		df_f_simple = gen_simple_table(port_f, port_f_net, use_net=False)
	print(df_f_simple)

	# %% Save structures to pickle files
	# pickle_dir = '../data'

	# df_stats.to_pickle(os.path.join(pickle_dir,fname))
	# df_stats.to_clipboard()

	print("Done")
