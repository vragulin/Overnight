# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:56:59 2022
Load new data (from James's list into a single dataframe)
@author: vragu
"""

#%% Load modules
import os
import pandas as pd
import pickle
import numpy as np
import config as cf

# Global variables - file locations
TICKER_FILE    = "elm_good_tkrs.pickle"
ALL_STOCKS_DFS = "elm_stocks_px_mcap.pickle"

#%% Load Good Elm Tickers from CSV File
def load_Elm_tickers(reload=False):
    
    ticker_fpath = os.path.join(cf.DATA_DIR, TICKER_FILE)
    if reload:
        """ load Bbg tickers from an CSV file """
        src_fbase = "Elm_Good_Tickers.csv"
        src_fpath = os.path.join(cf.DATA_DIR, src_fbase)
        
        # Read Excel file
        df = pd.read_csv(src_fpath)
             
        # Save tickers into a pickle file
        tickers = df['ticker'].values
        with open(ticker_fpath, "wb") as f:
            pickle.dump(tickers, f)
            
    else:
        # If reload is False, just load tickers from pickle file
        with open(ticker_fpath, "rb") as f:
            tickers = pickle.load(f)
            
    return tickers
   
#%% Test program for load_Elm_tickers
def test_load_Elm_tickers():
    
    # Load tickers form file
    tickers = load_Elm_tickers(reload=True)
    print(tickers[:5], tickers[-5:])
    
    
    
    # Load tickers from pickle
    tickers = None
    tickers = load_Elm_tickers(reload=False)
    print(tickers[:5], tickers[-5:])
    
    
#%% Collect data into a single dataframe
def compile_data():
    """ Build a data frame from csv files """
    tickers = load_Elm_tickers()

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):    #enumerate(['CIT']):   
        print(f"Processing {count}: {ticker}")
        df = pd.read_csv(os.path.join(cf.SIM_STK_DIR, f'{ticker}.csv'),
                         index_col='Date', parse_dates=True, dayfirst=True)
       
        # Calculte adjusted open and returns
        df['Factor']   = df['Close'] / df['Adj Close']
        df['Adj Open'] = df ['Open'] / df['Factor']
       
        # Drop unneeded columns to save memory
        df = df.drop(columns = ['Volume','$Vol', 'Factor'])
                    
        # # Calculate log returns - don't need them yet
        df['r_full'] = np.log(df['Adj Close']).diff()
        df['r_intr'] = np.log(df['Adj Close']) - np.log(df['Adj Open'])
        df['r_ovnt'] = df['r_full'] - df['r_intr']

        # Convert into long format table so it's easier to aggregate across stocks
        df1 = df.unstack().reset_index()
        df1.columns = ['Field', 'Date', 'Value']
        df1['Ticker'] = ticker
        
        # Rearrange columns to make it easier to read
        df1 = df1[['Ticker', 'Date', 'Field','Value']].dropna(subset=['Value']) 
        
        # Append the ticker to the large data frame
        if main_df.empty:
            main_df = df1
        else:
            main_df = pd.concat([main_df, df1]).reset_index(drop=True)
            
    print(f"Finished. Aggregate df has {len(main_df)} rows.")
    main_df.to_pickle(os.path.join(cf.DATA_DIR, ALL_STOCKS_DFS))
        
    return main_df
#%% Main program 
def main():
    
    test_load_tickers = False
    if test_load_tickers:
        test_load_Elm_tickers()
    
    df = compile_data()
    print("Done")
    
    return df

if __name__ == "__main__":
    main()
    
