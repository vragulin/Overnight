# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:50:29 2022

Simulate a strategy of going long/short portfolio of top/bottom deciles based on OBP

@author: vragu
"""

import datetime as dt
import os
import pandas as pd
import pickle
import numpy as np
from statsmodels.api import OLS, add_constant
import matplotlib.pyplot as plt


#%% Global variables - file locations
TICKER_FILE         = "data/sp500tickers.pickle"
ALL_STOCKS_DF       = "data/all_stocks_px_ret.pickle"
STOCK_DFS_DIR       = "stock_dfs"
MONTHLY_FIELD_DF    = "data/monthly_{field}.pickle"

#%% Load all stock panel data from pickle files
def load_data():
    
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)
    
    with open(ALL_STOCKS_DF, "rb") as f:
        df_all_stocks = pickle.load(f)

    return tickers, df_all_stocks


#%% Build a monthly dataframe of overnight returns
def load_monthly_df(field = 'r_ovnt', rebuild=False, save_df=True):
    """ Parameters:
            field   - name of a field that we want to load for every stock
            rebuild - if True, generate monthly o/n return from panel data
                     else, load from a pickle file
            save_df - save the resulting dataframe into a pickle
    """
    
    pickle_fname = MONTHLY_FIELD_DF.format(field=field)
    
    if not rebuild:
        # Load file from pickle
        with open(pickle_fname, "rb") as f:
            dfm = pickle.load(f)
    else:
        # Re-generate monthy on returns form panel data
        tickers, df_all_stocks = load_data()
        df1 = df_all_stocks.pivot_table(values='Value', index='Date', columns=['Ticker', 'Field'])
            
        # Extract Intraday Returns for all stocks
        df2 = df1.swaplevel(axis=1)[field]
        
        # Resample to monthly frequency
        dfm = df2.resample('M').sum()

        # Save new df into a pickle
        if save_df:
            with open(pickle_fname, "wb") as f:
                pickle.dump(dfm, f)
            
    return dfm
  
#%% Caculate Buy/Sell portfolios for each date

    
#%% Start of the main program
if __name__ == "__main__":
    
    # Generate dataframes of monthly overnight and intraday returns
    rebuild = False # Get data from pickle files
    df_o = load_monthly_df(field = 'r_ovnt', rebuild=rebuild)
    df_i = load_monthly_df(field = 'r_intr', rebuild=rebuild)
    df_f = df_o + df_i #full day return
    
    
    # Calculate the sorting parameter, for now just use (o/n - intra) window, later can do opb as in Lachance
    window  = 24 #window for the return calculation
    df_obp = (df_o-df_i).rolling(window).sum()
    
    # Generate top and bottom quantiles on the basis of the sorting parameter
    trade_pctiles  = [20, 80]  #Sell and buy  thresholds for shorts/longs portfolios
    
    # Generate positions
    #OBP thresholds long/short portfolios
    thresholds = pd.DataFrame(np.nan, index = df_obp.index, columns=['buy','sell', 'n_buys', 'n_sells'])
    #Positions in each stock at the close of a given date close
    positions  = pd.DataFrame(np.nan, index = df_obp.index, columns = df_obp.columns)
    
    for t, date in enumerate(df_obp.index):
        if t < window-1:
            continue
        
        row  = df_obp.loc[date,:].values
        non_zero_rets = row[row != 0]
        
        # Calculate thresholds for long/short portfolio
        t_sell, t_buy = np.percentile(non_zero_rets, trade_pctiles)
        
        # Calculate positions for all stocks
        positions.loc[date,:] = np.where(df_obp.loc[date,:]>=t_buy,1,0)
        positions.loc[date,:] = np.where(df_obp.loc[date,:]<=t_sell,-1, positions.loc[date,:])
        
        # Calculate the number of longs and shorts
        n_buys  = (positions.loc[date,:] ==  1).values.sum()
        n_sells = (positions.loc[date,:] == -1).values.sum()
        
        print(f"t={t}, t_buy={t_buy:.3f}, t_sell={t_sell:.3f}, n_buys={n_buys}, n_sells={n_sells}")

        thresholds.loc[date,:] = t_buy, t_sell, n_buys, n_sells
        
    # Calculate portfolio monthly and cumulative returns, with and w/o trans costs
    trx_costs  = 2 #one-way trading costs in bp
    capital    = 1000000  #initial capital
    
    port_cols  = ['r_l', 'r_s', 'r_ls', 'r_ix', 'r_ix_f']
    port_r     = pd.DataFrame(np.nan, index = df_obp.index, columns = port_cols)
    port_r_net = pd.DataFrame(np.nan, index = df_obp.index, columns = port_cols)
    
    # Convert exponential to simlpe returns, so we can aggregate the cross-section
    df_o_s = np.exp(df_o) -1
    df_i_s = np.exp(df_i) -1
    df_f_s = np.exp(df_f) -1
    
    # Populate dataframe with portfolio returns
    for t, date in enumerate(df_obp.index):
        if t <= window-1:
            continue
    
        # Extract position and return rows
        pos_row = positions.iloc[t-1,:]
        ret_row = df_o_s.iloc[t,:]

        # Identify long and short stocks
        long_mask  = ( pos_row ==  1  )
        short_mask = ( pos_row == -1 )
        
        # Gross return on the longs
        stock_r = pos_row * ret_row
        
        port_r.loc[date,'r_l' ]   = stock_r[long_mask].mean()
        port_r.loc[date,'r_s' ]   = stock_r[short_mask].mean()  #* Need to add risk-free return at some point
        port_r.loc[date,'r_ls']   = port_r.loc[date,'r_l'] + port_r.loc[date,'r_s']
        port_r.loc[date,'r_ix']   = df_o_s.iloc[t,:].mean()
        port_r.loc[date,'r_ix_f'] = df_f_s.iloc[t,:].mean()
              
        # Caclulate net returns - assume we trade 2x per day on each position
        # Position changes
        stock_r_net     = stock_r - trx_costs * 2 / 10000

        port_r_net.loc[date,'r_l' ] = stock_r_net[long_mask].mean()
        port_r_net.loc[date,'r_s' ] = stock_r_net[short_mask].mean()
        port_r_net.loc[date,'r_ls'] = port_r_net.loc[date,'r_l'] + port_r.loc[date,'r_s']
        port_r_net.loc[date,'r_ix'] = df_o_s.iloc[t,:].mean() - trx_costs * 2 / 10000
    
        # Index return - subtract initial transactions costs, otherwise assume no rebal
        if t == window:
            port_r_net.loc[date,'r_ix_f'] = df_f_s.iloc[t,:].mean() - trx_costs/10000
        else:
            port_r_net.loc[date,'r_ix_f'] = df_f_s.iloc[t,:].mean()       
        
    # Look at history (long/short/full) over Lachance sample period and since 2015
    # Cumulative returns
    port_r_cum     = (1+port_r).cumprod()
    port_r_net_cum = (1+port_r_net).cumprod()

    
    #%% Plots - paper sample
    cutoff_date = '2015-01-01'
    
    legend_list = ['long_on','short_on','l/s_on', 'index_on', 'index_full']
    # _ = np.log(port_r_cum.loc[:cutoff_date,:]).plot(title = 'Gross Log O/N Returns - Before 2015')
    # plt.legend(legend_list)
    # plt.show()
    
    _ = np.log(port_r_net_cum.loc[:cutoff_date,:]).plot(title = 'Net Log O/N Cum P&L - Before 2015')
    plt.legend(legend_list)
    plt.show()
    
    #%% Plot - recent data
    
    port_r_cum1      = (1+port_r.loc[cutoff_date:,:]).cumprod()
    port_r_net_cum1 = (1+port_r_net.loc[cutoff_date:,:]).cumprod()
    
    # _ = np.log(port_r_cum1).plot(title = 'Gross Log O/N Returns - Since 2015')
    # plt.legend(legend_list)
    # plt.show()
    
    _ = np.log(port_r_net_cum1).plot(title = 'Net Log O/N Cum P&L - Since 2015')
    plt.legend(legend_list)
    plt.show()
    
    print("Done")
    