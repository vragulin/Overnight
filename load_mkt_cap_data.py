# -*- coding: utf-8 -*-
"""
Created on Thu May 19 18:46:10 2022
Load fundamental stock data from yfinance
@author: vragu
"""

import pandas as pd
import os 
import yfinance as yf
import numpy as np
import concurrent.futures
import pickle

DATA_DIR = "../data"

#%% Load Market Cap and other data - single threaded function, a bit slow
def load_mcap(tickers=None, fpath=None, reload=True, save=True):
    """ Load the latest market cap for a list of stocks
        Params:
            tickers - list of tickers (or another iterable)
            fpath   - path of the pickle file with the data for saving and loading
            reload  - if True, reload from yfinance, else load from pickle
            save    - save output into pickle file specified by fpath
            
        Return: pandas dataframe with market caps for each stock
    """
    # Create a yfinance object with the list of tickers
    t = yf.Tickers(list(tickers))

    mcap = pd.DataFrame(0, index=tickers, columns = ['name', 'mkt_cap', 'last', 'shrs_out'])

    for ticker in tickers[:5]:
        print(ticker)
        mcap.loc[ticker, 'name']    = t.tickers[ticker].info['shortName']        
        mcap.loc[ticker, 'mkt_cap'] = t.tickers[ticker].info['marketCap']
        mcap.loc[ticker, 'last']    = t.tickers[ticker].info['regularMarketPrice']
        
    mcap['shrs_out'] = np.where(mcap['last'] != 0, mcap['mkt_cap'] / mcap['last'],np.nan)

    return mcap

#%% Same as above, but use multithreading
def load_mcap_multhread(tickers=None, fpath=None, reload=True, save=True):
    """ Load the latest market cap for a list of stocks
        Params:
            tickers - list of tickers (or another iterable)
            fpath   - path of the pickle file with the data for saving and loading
            reload  - if True, reload from yfinance, else load from pickle
            save    - save output into pickle file specified by fpath
            
        Return: pandas dataframe with market caps for each stock
    """

    # If update not required, we load data from a pickle
    if not reload:
        if fpath:
            with open(fpath, "rb") as f:
                mcap = pickle.load(f)
        else:
            print("Could not read data, file name not specified")
            return None
    else:
        # Create a yfinance object with the list of tickers
        t = yf.Tickers(list(tickers))
    
        mcap = pd.DataFrame(0, index=tickers, columns = ['name', 'mkt_cap', 'last', 'shrs_out'])
        
        def get_data_one_symbol(sym):
            name    = None
            mkt_cap = None
            last    = None
            try:
                name    = t.tickers[sym].info['shortName']        
                mkt_cap = t.tickers[sym].info['marketCap']
                last    = t.tickers[sym].info['regularMarketPrice']
            except Exception:
                pass
            return (sym, name, mkt_cap, last)
    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(get_data_one_symbol, sym): sym for sym in tickers}
            for future in concurrent.futures.as_completed(futures):
                sym, name, mkt_cap, last = future.result()
                if name:
                    print(f'{sym}: {name}: MktCap = {mkt_cap}, Last = {last}')
                    mcap.loc[sym,['name','mkt_cap','last']] = name, mkt_cap, last
            
        mcap['shrs_out'] = np.where(mcap['last'] != 0, mcap['mkt_cap'] / mcap['last'],np.nan)
    
        # If required, save the structure
        if save:
            if fpath:
                with open(fpath, "wb") as f:
                    pickle.dump(mcap, f)
            else:
                print("Count not save, file name not specified")
                
    return mcap

#%% Start of  Main
# =============================================================================
if __name__ == "__main__":
    
    tickers = pd.read_pickle(os.path.join(DATA_DIR,"elm_good_tkrs.pickle"))
    
    fpath = os.path.join(DATA_DIR, "mkt_cap2.pickle")
    mcap = load_mcap_multhread(tickers = tickers, fpath=fpath)
    
    print(mcap)
    print("Done")