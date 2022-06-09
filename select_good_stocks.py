# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:37:34 2022

Filter out stocks with sufficiently good data to be included in simulation
Save file with metadata (
    * which conditions True/False for each stock
    * what are start/end date of good data

@author: vragu
"""

import pandas as pd
import os
import numpy as np
import pickle
#import json

import config as cf

#%% Initialize parameters and load data needed for the test
def load_context():
    """ Load parameters and data needed for the tests """
    
    context = {}

    # =============================================================================
    # Parameters
    # =============================================================================
    # Average volume test
    context['avg_vol_win']     = 22
    context['min_avg_volume']  = 1e7
    
    # Market cap test
    context['min_mkt_cap']     = 4e9
    
    # Trading days test
    context['max_miss_day']    = 0
    
    # Min close price
    context['check_min_px']    = False
    context['min_px']          = 5
    
    # Max/Min returns test
    context['check_ret_range'] = False
    context['max_ret']         = 1.5   #(150% rally in 1 day)
    context['min_ret']         = -0.6   #(60% loss in 1 day)

    # Window over which there should be no "bad" events
    context['good_px_win']     = 22   
    
    # Minimum number of good days for the stock to be used in the simulation
    context['min_good_days']     = 760  # 3 years   
    
    # Load trading days calendar and S&P factors
    trd_days     = pd.read_pickle(os.path.join(cf.DATA_DIR, "trading_days.pickle")).set_index('Date')
    spx_factors  = pd.read_pickle(os.path.join(cf.DATA_DIR, "SPX_DiscFactors.pickle"))[['DiscFactor']]

    df = trd_days.join(spx_factors, how='left')

    # Merge days and factors into a single dataframe
    context['trd_days'] = df

    return context

#%% Validate one stock vs. constraints
def validate_one_stock(ticker, context):
    """ Check if a ticker satisfies all the constraints for validation
        Constraints are as follows:
            * Avg 20-day volume $10mm (adj by SPX)
            * Mkt cap > $4bio (discounted by SPX)
            * Stock has open/close prices 
            * Close prices are above $5  - don't use
            * O/N or intraday return >250% or <-60% - don't use for the moment
            * All of the above is true for every day during the prior 22-day window period
    """
    
    # =============================================================================
    # Load data
    # =============================================================================
    price_file = os.path.join(cf.MERGED_DIR, f"{ticker}.csv")
    df_px = pd.read_csv(price_file, index_col = 'Date', parse_dates = True)
    if (df_px is None) or (len(df_px) == 0):
        print(f"{ticker} : No price data")
        return None, None
    
    # Drop unnecessary columns
    df_px.drop(['High','Low', 'Split Adj Close', 'shrsOut', 'Close ref date', 
                'Split Adj Close ref date', 'Ref Date'], axis=1, inplace=True)
    
    # Set up a dataframe with test results for each date - for now all False
    df_test = context['trd_days']
    cols = ['avg_vol', 'mkt_cap', 'no_nans', 'no_low_px', 'no_big_moves', 'all', 'all_window']
    for col in cols:
        df_test[col] = False
    
    # Merge df_test dataframe
    df = df_test.join(df_px, how='left')
       
    # =============================================================================
    # Run tests
    # =============================================================================
    # Average daily volume
    df['$Vol'] = df['Close'] * df['Volume']
    df['Avg $Vol'] = df['$Vol'].rolling(context['avg_vol_win'],min_periods=1).mean()
    df['avg_vol'] = (df['Avg $Vol'] / df['DiscFactor']) >= context['min_avg_volume']


    # Market cap
    df['mkt_cap'] = (df['Mkt Cap'] / df['DiscFactor'])>= context['min_mkt_cap']
    
    # Stock has open and close prices 
    missed   = np.isnan((df['Close'] + df['Open'])) #If either is NaN, the sum will be also
    df['no_nans'] = ~missed
    #n_missed = missed.rolling(context['no_miss_px_win'],min_periods=1).sum()
    #df['no_miss_days']  = n_missed <= context['max_miss_day']
       
    # Close prices below threshold ($5)
    if context['check_min_px']:
        df['no_low_px'] = df['Close'] >= context['min_px']
    else:
        df['no_low_px'] = True
    
    # O/N or intraday return outside of the range at any point during the window
    if context['check_ret_range']:
        ovnt_ret = df['Close'].pct_change()
        intr_ret = df['Close'] / df['Open'] - 1
        max_ret, min_ret  = context['max_ret'], context['min_ret']
        jump = (ovnt_ret > max_ret) | (ovnt_ret < min_ret) | (intr_ret > max_ret) | (intr_ret < min_ret)
        df['no_big_moves'] = ~jump
    else:
        df['no_big_moves'] = True
        
    # Check that we passed all the tests
    df['all'] = df[['avg_vol', 'mkt_cap', 'no_nans', 'no_low_px', 'no_big_moves']].all(axis=1)
    
    # Check that there were no issues during the prior window (1mo)
    df['all_window'] = df['all'].rolling(context['good_px_win'],min_periods=1).min().astype(bool)
    df_good = df[df.all_window]
    
    # Collect metadata that will help decide whether to keep or remove the stock from consideration
    # This will be implemented later
    stats = {}
    stats['days']       = len(df_px[df_px['Close'] > 0])
    stats['good_days']  = len(df_good)
    if stats['good_days'] > 0 :
        stats['start']      = min(df_good.index)
        stats['end']        = max(df_good.index)
    else:
        stats['start']      = None
        stats['end']        = None
    
    # Include the stock into simulation?
    stats['use']        = stats['good_days'] >= context['min_good_days']
    
    return df, stats

def select_stocks_for_sim(context):
    """ Process all stocks, save stats for all stocks
        Return a dataframe with statistic for all stocks    
    """
    
#%% Entry point
if __name__ == "__main__":

    # Load context (necessary params and data)
    context = load_context()

    # For every price file in the merged directory
    # Extract the ticker name
    # Run the analysis for one ticker
    # Save results (probably both in a dataframe and a file in case the run interrupts)
    # Maybe call the directly for run results "stock_metadata"

    # Ensure that the directory for data files exists
    if not os.path.exists(cf.HSTAT_DIR):
        os.makedirs(cf.HSTAT_DIR)
    
    if not os.path.exists(cf.SIM_STK_DIR):
        os.makedirs(cf.SIM_STK_DIR)
    
    # Loop over all stocks for which we have annual reports (i.e. shrsOutstanding data)
    results = pd.DataFrame(columns=['ticker', 'days','good_days', 'start', 'end', 'use'])
    cols_to_save = ['Open', 'Close', 'Adj Close', 'Volume', 'Mkt Cap', '$Vol', 'all_window']
    for i, file in enumerate(os.listdir(cf.MERGED_DIR)): #enumerate(['AMC.csv']):
        ticker = os.path.splitext(file)[0]

        # Check if ticker has already been processed
        dest_file = os.path.join(cf.SIM_STK_DIR, f'{ticker}.csv')
        stat_file = os.path.join(cf.HSTAT_DIR, f'{ticker}.pickle')
        if os.path.exists(stat_file):
            print(f'{i} : {ticker} : already done')
            continue
        else:        
            # Process the ticker and save the good data
            print(f'{i} : {ticker}', end="")
            df, stats = validate_one_stock(ticker, context)

            # Save the results in a dataframe
            if stats is None:
                stats = {}
                stats['use'] = False
                
            stats['ticker'] = ticker
            results = results.append(stats, ignore_index=True)
            with open(stat_file, "wb") as f:
                pickle.dump(stats, f)

            # If the ticker passes all criteria, save into directory
            if stats['use']:
                print(" ... keeping.")
                print(stats)
                try:
                    df[cols_to_save].to_csv(dest_file)
                except AttributeError as e:
                    print(f"No data for {ticker}. Error Msg: ", e)
            else:
                print("... dropping.")
    
    print(results.head())
    results.to_clipboard()
    print("Done")
    
        
        
        
    
    


