# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:50:29 2022

Find stocks with the largest recent OBP returns (overnight vs. intraday)
Largely based on simulator2

@author: vragu
"""

#%% Load modules
#import os
import pandas as pd
import pickle
#import numpy as np
import datetime as dt
#import os

#%% Global variables - file locations
TICKER_FILE         = "../data/elm_good_tkrs.pickle"
ALL_STOCKS_DF       = "../data/elm_stocks_px_mcap.pickle"
STOCK_DFS_DIR       = "../stock_dfs"
MONTHLY_FIELD_DF    = "../data/elm_monthly_{field}.pickle"
IDX_WEIGHT_FILE     = "../data/cap1_idx_weights.pickle"

TD_PER_MONTH        = 21.6 #numbe of trading days per month

#%% Initialize Simulation Parameters
def init_sim_params():
    """ Initialize Sim Params, return dictionary"""
    params = {}
    params['rebuild'      ] = False    #Regenerate stock return files
    params['window'       ] = 42      #window in months for the return calculation
    params['trx_costs'    ] = 0.5     #one-way trading costs in bp
    params['borrow_fee'   ] = 25      #borrow fee on the shorts, in bp per annum
    params['capital'      ] = 1       #initial capital
    params['trade_pctiles'] = [20, 80] #Sell and buy  thresholds for shorts/longs portfolios
    params['gross'        ] = False    #Whether to producec graphs and reports for gross or net returns
    params['max_weight'   ] = None     #Max weight in a portfolio as a multiple of equal-weights
    params['min_weight'   ] = None     #Min weight 
    params['short_gap'    ] = None     #If we exclude days with a long gap: True - Exclude, False - Only long gaps, None-ignore

    # Load index weights for the stock universe (we assume that our positions are proportional to these weights)
    # These can be thought of as market caps (which are re-scaled that total weight for both long and short portfoio
    # adds up to 1)
    params['idx_weights'] = pd.read_pickle(IDX_WEIGHT_FILE).astype(float)
    return params

#%% Load all stock panel data from pickle files
def load_data():
    
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)
    
    with open(ALL_STOCKS_DF, "rb") as f:
        df_all_stocks = pickle.load(f)

    return tickers, df_all_stocks


#%% Build a monthly dataframe of overnight returns
def load_monthly_df(field = 'r_ovnt', rebuild=False, save_df=True, short_gap=None):
    """ Parameters:
            field   - name of a field that we want to load for every stock
            rebuild - if True, generate monthly o/n return from panel data
                     else, load from a pickle file
            save_df - save the resulting dataframe into a pickle
            short_gap - if True - only include days for which t-1 was also a trading day
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
        #df1 = df_all_stocks.pivot_table(values='Value', index='Date', columns=['Ticker', 'Field'])
        df1 = df_all_stocks.pivot(values='Value', index='Date', columns=['Ticker', 'Field'])
                    
        # Extract Intraday Returns for all stocks
        df2 = df1.swaplevel(axis=1)[field].astype(float)
        
        # Mask out returns with long gaps
        if short_gap is not None:
            df2['_date_'] = df2.index
            df2['gap'] = df2['_date_'].diff() / dt.timedelta(days=1)
            if short_gap == True:   #Only consider days with short gap
                df2[df2['gap']>1] = 0
            else:
                df2[df2['gap']==1] = 0
            df2.drop(['_date_', 'gap'],axis=1, inplace=True)  # Drop columns that we don't need anymore
            

        # Resample to monthly frequency
        dfm = df2.resample('M').sum()

        # Save new df into a pickle
        if save_df:
            dfm.to_pickle(pickle_fname)
            
    return dfm

#%% Start of the main program
if __name__ == "__main__":
    
    # Initialize simulation parameters
    sim = init_sim_params()
    
    # Generate dataframes of monthly overnight and intraday returns
    df_o = load_monthly_df(field = 'r_ovnt', rebuild=sim['rebuild'], save_df=True, short_gap=sim['short_gap'])
    df_i = load_monthly_df(field = 'r_intr', rebuild=sim['rebuild'], save_df=True, short_gap=sim['short_gap'])
   
    
    # Calculate the sorting parameter, for now just use (o/n - intra) window, later can do opb as in Lachance
    df_obp = (df_o-df_i).rolling(sim['window']).sum()
    
    # Extract the latest returns
    rets = df_obp.iloc[-1,:]
    
    # Print highest and lowest
    rets.sort_values(ascending=False, inplace=True)   
    print(rets.head())
    
    rets.to_clipboard()

    