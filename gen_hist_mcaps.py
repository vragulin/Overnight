# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:24:14 2022

Generate a dataframe of historical market caps, use monthly frequency
Also, filtered file, which zeroes out stocks with low mkt cap (< threshold * SPY)

@author: vragu
"""

import pandas as pd
import os 
import numpy as np
import pickle

from an_pos_mkt_caps import load_field_all_stocks

DATA_DIR = "../data"
MC_FILE  = "mkt_caps_over_time.pickle"

#%% Generate historical market cap proxies
def gen_hist_mcap(file=None):
    
    # Load file with end-of-period market caps
    f_mc = 'mcap_updated.csv'
    mcap_last = pd.read_csv(os.path.join(DATA_DIR,f_mc),index_col="ticker", thousands=',')

    # Generate a market cap proxy for every month-end
    # Scale over time by unadjusted closes (not adjusted closes)
    # Unadjusted closes incorporate splits (in Yahoo, but not alpha-vantage), but not dividends
    reload=False # Generate data from individual stock files if True/else load from picle
    tickers = mcap_last.index
    f_cls = os.path.join(DATA_DIR, "all_closes.p")
    
    closes = load_field_all_stocks(tickers=tickers,field='Close',fname=f_cls, reload=reload, save=reload)
    closes_m = closes.fillna(method='ffill').resample('M').last()
    
    # Multiply closes by number of shares to get a series of market caps
    mc = closes_m * mcap_last['shrs_out']
    
    # Save into a pickle file
    if file is not None:
        mc.to_pickle(file)
        
    return mc

#%% Main program
def main():

    mcap = gen_hist_mcap(file=os.path.join(DATA_DIR, MC_FILE))
    
    print(mcap.head())
    print("Done")
    
    
if __name__ == "__main__":
    main()