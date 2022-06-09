# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 09:41:46 2022
Process stock files to generate additional useful series: split-adjusted close, market cap etc.
For each stock generate a dataframe that has a data point for every trading day

@author: vragu
"""

import numpy as np
import pandas as pd
import os
import calc_adj_close as cac

import config as cf

#%% Generate a series of market cap data
def gen_mkt_cap_hist(prices, shs_out):
    """
    Generates a periodic (D/M/Q) series of market caps using the following logic:
     * for every date t take the nearest to t (on ref_date) number of shares.
     * caclulate mcap as shrsOut(ref_date) * Close(ref_date) / split_adj_Close(ref_date) * split_adj_close(t)
     
    Parameters
    ----------
    prices : dataframe with periodic (D/M/Q etc.) historices of Closing and Split-Adj closing prices
    shs_out : dataframe with numbers of shares outstanding on different reporting dates

    Returns
    -------
    df : dataframe containing the input series + a full series of market caps on each date

    """
    # Check that the shares_out input is not empty
    if (shs_out is None) or (len(shs_out) == 0):
        print("Error: no data for sharesOutstanding. Unable to calc market cap.")
        return None
    
    # Sort shs_out by index in ascending order
    shs_out.sort_index(inplace=True)
    shs_out['Ref Date'] = shs_out.index
    
    # Capture Close and Split-Adj Close on reporting dates
    shs_out1 = pd.merge_asof(shs_out, prices, left_index=True, right_index=True, direction='nearest')
    
    # Merge sh_out with prices, match by nearest date
    df = pd.merge_asof(prices, shs_out1[['shrsOut','Close','Split Adj Close','Ref Date']], 
                       left_index=True, right_index=True, direction='nearest', suffixes=[""," ref date"])
    
    # Calculate market cap and drop unneeded columns
    # df['Mkt Cap'] = df['shrsOut'] * df['Close ref date'] / df['Split Adj Close ref date'] \
    #                                 * df['Split Adj Close']
    df['Mkt Cap'] = df['shrsOut'] * df['Split Adj Close']
    
    return df

#%% Process data from different files for one stock
def process_one_stock(ticker):
    """ Load stock data from csv files.  
        Clean out bad data
        Create all series necessary for analysis (e.g. market cap)
        
        Parameters:
            * ticker : ticker (e.g. AAPL)
            
        Return: 
            * dataframe with the data
            * dataframe is also saved into a directory as a csv file
    """
       
    # =============================================================================
    # Load stock data: price, splits and dividends
    # =============================================================================
    price_file = os.path.join(cf.STOCK_DIR, f"{ticker}.csv")
    df = pd.read_csv(price_file, index_col = 'Date', parse_dates = True)
    if (df is None) or (len(df) == 0):
        print(f"{ticker} : No price data")
        return None

    # If Close/Open/Adj Close price or volume on a given day are zero, replace with NaN and drop
    for field in ['Close', 'Open', 'Adj Close']:
        df[field] = np.where((df[field] == 0) | (df['Volume'] == 0), np.nan, df[field])
    df.dropna(subset=['Open','Close','Adj Close'],inplace=True)

    # Process splits
    split_file = os.path.join(cf.SPLIT_DIR, f"{ticker}.csv")
    if os.path.exists(split_file):
        df_split = pd.read_csv(split_file, index_col = 'Date', parse_dates = True)    
                
        # Generate split-adjusted closing prices
        df_adj_close = cac.calc_adj_closes(df['Close'], splits=df_split['Stock Splits'], source='eod_hd')
        df['Split Adj Close'] = df_adj_close['Close']
    else:
        df['Split Adj Close'] = df['Close']
  
    # Generate a series of historical market caps
    shs_out_file = os.path.join(cf.SHRS_OUT_DIR, f"{ticker}.csv")
    df_shs_out = pd.read_csv(shs_out_file, index_col = "Date", parse_dates=True)
    
    # =============================================================================
    # Need to write a function to generate a historical series of market caps!!!
    # =============================================================================
    df_out = gen_mkt_cap_hist(df, df_shs_out)
    
    return df_out
    
#%% Entry Point
if __name__ == "__main__":
    
    # Ensure that the directory for data files exists
    if not os.path.exists(cf.MERGED_DIR):
        os.makedirs(cf.MERGED_DIR)
        
    # Loop over all stocks for which we have annual reports (i.e. shrsOutstanding data)
    for i, file in enumerate(os.listdir(cf.SHRS_OUT_DIR)): #enumerate(['CIT.csv']):
        ticker = os.path.splitext(file)[0]

        # Check if ticker has already been processed
        dest_file = os.path.join(cf.MERGED_DIR, f'{ticker}.csv')
        if os.path.exists(dest_file):
            print(f'{i} : {ticker} : already done')
            continue
        else:        
            # Process the ticker
            print(f'{i} : {ticker}')
            df = process_one_stock(ticker)
            try:
                df.to_csv(dest_file)
            except AttributeError as e:
                print(f"No data for {ticker}. Error Msg: ", e)
    
    print("Done")
    
    
