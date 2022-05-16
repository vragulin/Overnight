# -*- coding: utf-8 -*-
"""

Created on Sun May 15 12:32:56 2022
Replicate Figure 1 (Fama-McBeth regressions) from the Lachance paper
@author: vragu

"""

#import datetime as dt
#import os
#import pandas as pd
import pickle
import numpy as np
from statsmodels.api import OLS, add_constant
import matplotlib.pyplot as plt

#%% Global variables - file locations
TICKER_FILE    = "data/sp500tickers.pickle"
ALL_STOCKS_DFS = "data/all_stocks_px_ret.pickle"
STOCK_DFS_DIR  = "stock_dfs"
MONTHLY_R_ON_DFS = "data/monthly_r_on.pickle"

# Load all stock panel data from pickle files
def load_data():
    
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)
    
    with open(ALL_STOCKS_DFS, "rb") as f:
        df_all_stocks = pickle.load(f)

    return tickers, df_all_stocks

#%% Build a monthly dataframe of overnight returns
def load_monthly_df(rebuild=False, save_df=True):
    """ Parameters:
            rebuild - if True, generate monthly o/n return from panel data
                     else, load from a pickle file
            save_df - save the resulting dataframe into a pickle
    """
    
    if not rebuild:
        # Load file from pickle
        with open(MONTHLY_R_ON_DFS, "rb") as f:
            dfm = pickle.load(f)
    else:
        # Re-generate monthy on returns form panel data
        tickers, df_all_stocks = load_data()
        df1 = df_all_stocks.pivot_table(values='Value', index='Date', columns=['Ticker', 'Field'])
            
        # Extract Intraday Returns for all stocks
        df2 = df1.swaplevel(axis=1)['r_ovnt']
        
        # Resample to monthly frequency
        dfm = df2.resample('M').sum()

        # Save new df into a pickle
        if save_df:
            with open(MONTHLY_R_ON_DFS, "wb") as f:
                pickle.dump(dfm, f)
            
    return dfm

#%% Run Fama-MacBeth regression for one month
def FM_reg_one_date(t, lag, dfm, cumret = False):
    """ Parameters:
        This function produces a time series of betas - one for each month
        
        t      - int, index of the date of the dependent return (LHS)
        lag    - lag (k) between dependent and explanatory return
        dfm    - data frame of monthly returns
        cumret - if True we take average return from t-k to t-1, if False we regress on t-k return
    """
    
    # Check that we have sufficient data for the lag
    if (t < lag) or (t >= len(dfm)) or (lag >= len(dfm)):
        raise ValueError(f"Error: Insufficient data. t={t}, lag={lag}, len(df)={len(dfm)}")
    
    # Set up regression variables
    Y = dfm.iloc[t,:].values
    
    if (not cumret) or (lag == 1) :
        X = dfm.iloc[t-lag,:].values
    else:
        X = dfm.iloc[t-lag:t,:].mean(axis=0).values
    
    X = add_constant(X)

    # Fit the model
    results = OLS(Y,X).fit()

    return results

#%% Run Fama-MacBeth regression for the entire sample
def FM_reg_all_dates(lag, dfm, cumret = False):
    """ Parameters:
        This function imposes a restriction that beta is the same across the entire period and estimates it
        
        lag    - lag (k) between dependent and explanatory return
        dfm    - data frame of monthly returns
        cumret - if True we take average return from t-k to t-1, if False we regress on t-k return
    """
    
    # Check that we have sufficient data for the lag
    if lag >= len(dfm):
        raise ValueError(f"Error: Insufficient data. lag={lag}, len(df)={len(dfm)}")
    
    # Set up regression variables
    Y = dfm.iloc[lag:,:].values.ravel()
    
    if (not cumret) or (lag == 1) :
        X = dfm.iloc[:-lag,:].values.ravel()
    else:
        dfm1 = dfm.rolling(lag).mean()
        X = dfm1.iloc[lag-1:-1,:].values.ravel()
    
    X = add_constant(X)

    # Fit the model
    results = OLS(Y,X).fit()

    return results
    
    
#%% Main
if __name__ == "__main__":
   
    # Generate a data frame of monthly overnight returns
    rebuild = False #Do not rebuild a databse of monthly returns, fetch from pickle
    dfm = load_monthly_df(rebuild=rebuild)

    dfm_before = dfm['1995-01-01':'2014-12-31']
    dfm_after  = dfm['2014-12-31':]
    
    # Run regressions    
    df = dfm_after
    all_stats = []
    
    for df in [dfm_before, dfm_after]:
        
        tdict  = {}
        for cumret in [False, True]:
            
            tstats = []
            for lag in range(1,61):
    
                # Run a single regression across all months
                results = FM_reg_all_dates(lag,df, cumret=cumret)
                tstat = results.tvalues[1]
                tstats.append(tstat)
                print(f"Lag = {lag}, tstat = {tstat}")
                
            tdict[cumret] = tstats
            
        all_stats.append(tdict)
    
    #%% Plot line for both individual and cumulative returns
    # Before publication
    tdict = all_stats[0]
    plt.bar( np.arange(len(tdict[False])),tdict[False],color='blue', alpha=0.5)
    plt.plot(np.arange(len(tdict[True])),tdict[True]  ,color='blue')
    
    # After publicatio
    tdict = all_stats[1]
    plt.bar( np.arange(len(tdict[False])),tdict[False],color='red', alpha=0.5)
    plt.plot(np.arange(len(tdict[True])),tdict[True]  ,color='red')
    
    plt.legend(['Before: Avg ret (months 1:k)', \
                'After: Avg ret (months 1:k)',\
                'Before: Lag k monthly o/n ret',\
                'After:  Lag k monthly o/n ret'],loc='upper right', fontsize='small')
    plt.suptitle("Regressions of Monthly Overnight Returns on Lagged Values: t-stats",fontsize='medium')
    plt.title("Before 2015 (1995-2014) and after 2015", fontsize='small')
    plt.show()
    
    print("Done")
    
    