# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:22:45 2022
Check yahoo data vs. alpha-vantage
@author: vragu
"""


import pandas as pd
import pickle
import os
import numpy as np

# Global variables
TICKER_FILE    = "../data/sp500tickers.pickle"
YAHOO_DIR  = "../stock_dfs_w_div_splits_20220525"
AV_DIR     = "../stock_dfs"



def check_data_one_ticker(ticker, tol = 0.5):
    """ Compare yahoo and alpha-vantage data, ensure it's consistent"""
    
    # Load yanoo data
    fname = os.path.join(YAHOO_DIR, f"{ticker}.csv")
    df1 = pd.read_csv(fname, index_col="Date", parse_dates=True)
    
    # Load av data
    fname = os.path.join(AV_DIR, f"{ticker}.csv")
    df2 = pd.read_csv(fname, index_col="Date", parse_dates=True)
    
    # Check if the two series diverge over tolerance
    df = df1[['Adj Close']].join(df2[['Adj Close']],how='left', lsuffix='_yh', rsuffix='_av').dropna()

    df['pdiff']    = df['Adj Close_yh'] / df['Adj Close_av'] - 1
    df['over_tol'] = np.abs(df['pdiff']) > tol
    
    return df['over_tol'].any()

#%% Start of the main program
def main():
    # Load tickers
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)
    
    # Loop over tickers
    for ticker in ['DD']:  # tickers:
        
        res = check_data_one_ticker(ticker)
        if not res:
            print(ticker)        
        else:
            print(f"{ticker} has different data")
            
    print("\nDone")
    

if __name__ == "__main__":
    main()
    