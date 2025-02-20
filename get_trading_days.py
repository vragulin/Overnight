# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:44:30 2022
Build a list of trading dates, which will be the index for all dataframes

@author: vragu
"""

import pandas as pd
import os

#%% Global variables
SHRS_DFS_DIR = "../stock_dfs"
DATA_DIR     = "../data"

#%% Generate trading dates
def gen_trading_dates(ref_stocks, start=None, end=None, verbose=False):
    """ Generate a list of valid trading dates by taking the union of trade dates
        for a list of reference 'large and liquid' stocks. """
        
    " Initialize a dataframe what will hold all data"
    df = None
    
    for stk in ref_stocks:
        
        print(stk)
        df_stk = pd.read_csv(os.path.join(SHRS_DFS_DIR, f"{stk}.csv"),
                             index_col = "Date", parse_dates=True)
        df_stk.rename(columns = {"Adj Close":stk}, inplace=True)
        
        
        if df is not None:
            df = df.join(df_stk[[stk]], how="outer")
        else:
            df = df_stk[[stk]]

    # Only keep dates within desired range
    df = df[start:end]
    
    # Find dates for which we don have data for any stocks
    df.dropna(how="all", inplace=True)
    
    if verbose:
        print(df.head())
        print(df.tail())
    
    df.reset_index(inplace=True)
    
    return(df[['Date']])

#%% Entry Point
if __name__ == "__main__":

    ref_stocks = ["MSFT", "IBM", "JPM", "C", "AAPL", "DUK"]    
    # start = '1993-01-01'
    # end   = '2022-06-01'
    
    df = gen_trading_dates(ref_stocks, verbose=True)
    
    # Save to data directory
    df.to_pickle(os.path.join(DATA_DIR, "trading_days.pickle"))
    df.to_csv(os.path.join(DATA_DIR, "trading_days.csv"))
