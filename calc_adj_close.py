# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:41:46 2022
Calculate my own adjusted closing price from unadj close, div and splits
@author: vragu
"""

import pandas as pd
import os
#import yfinance as yf

#%% Calculate my own adjusted close for a stock
def parse_eod_hd_splits(splits):
    """ 
    Parse a dataframe with splits data in eod_hd format to default format 
    EOD Hist Data gives splits as a string ['7.00 / 1.00'], default format 
    is just a multiplier.
    
    Parameters:
    -----------
    splits: pandas series with index date and column of slpits as 'xx/yy' strings.
    
    Returns
    -------
    df: dataframe with dates and float multipliers """
        
    # Separate the split string into a numerator and a denominator
    df = splits.str.split("/", n=1, expand=True)
    df.rename(columns={0:"Numerator", 1:"Denominator"}, inplace=True)
    df["Split String"] = splits
    df["Split Factor"] = df["Numerator"].astype(float) / df["Denominator"].astype(float)
    
    return df

def calc_adj_closes(closes, dividends=None, splits=None, source='default',
                    adj_div_for_splits = False):
    """
    Parameters
    ----------
    close : pandas series with daily frequency (later may be able to adapt for other frequencies)
    dividents : panda series with dividend amounts and dates, can be either dense (for all dates)
                or sparse (without zeros), if None - don't adjust
    splits : pandas series with splits dates and rates, dense or sparse, if None - don't adjust
    source : name of vendor, which determines the format of the "spit" dataframe.  In practice,
             only 2 values have been implemented: 
                 * 'default' - where values are floats (e.g. 7.0 for a 7-to-1 splits)
                 * 'eod_hd' - from (eod hist data) where splits are given as strings ['7.00/1.00']
    
    adj_div_for_splits : if dividends are not already adjusted for splits (i.e. are based off Close not Adj Close),
                         then perform this adjustment.  It's not necessary for Yahoo or EODHD, since divs there 
                         are already split-adjusted.
    
    Returns
    -------
    df: dataframe with adjusted closes prices for same dates as closes, splits factors and div factors
    """

    
    # Extract start and end data of the stock series
    start = min(closes.index)
    end   = max(closes.index)

    # Create a copy of dividends
    if dividends is not None:
        
        adj_dividends = dividends.copy()
        adj_dividends.name = "Dividends"
        # Only consider dividends within the relevant date range
        adj_dividends = adj_dividends[(adj_dividends.index >= start) & (adj_dividends.index <= end)]
    
    # Adjust for splits
    df = pd.DataFrame(closes).rename(columns={closes.name: "Close"})
    
    
    if splits is not None:
        if source == "eod_hd":
            splits1 = parse_eod_hd_splits(splits)['Split Factor']
        else:
            splits1 = splits.copy()
            
        # Only consider splits within the relevant date range
        splits1 = splits1[ (splits1.index >= start) & (splits1.index<=end) ]
        
        df = df.join(splits1[splits1>0], how='outer').sort_index(ascending=False)\
                    .rename(columns={splits1.name: "Splits"})
        split_coef = df['Splits'].shift(1).fillna(1).cumprod()
        df['Close'] = df['Close'] / split_coef
        
        if dividends is not None:
            if adj_div_for_splits:
                # If dividends are not already split-adjusted (in Yahoo and EOD HD they are, so this is not needed)
                adj_dividends = adj_dividends / split_coef
                
            adj_dividends.name = "Dividends"         

        df['Split_Factors'] = split_coef                
        
    if dividends is not None:
        df = df.join(adj_dividends[adj_dividends>0], how='outer').sort_index(ascending=False)\
                    .rename(columns={adj_dividends.name: "Dividends"})
                    
        div_coef = (1 - df['Dividends'] / df.shift(-1)['Close']).shift(1).fillna(1).cumprod()
        df['Close'] = df['Close'] * div_coef
        
        
        df['Div_Factors']   = div_coef
    
    return df.sort_index()

def build_test_file():
    
    df = pd.DataFrame({'Date':         pd.date_range('2022-05-16', periods=5),
                       'Close':        [30, 15, 11, 12, 6],
                       'Dividends':    [0 ,  0,  5,  0, 0],
                       'Stock Splits': [0,   2,  0,  0, 2]
        })
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)

    return df

#%% Main program - test 
if __name__ == "__main__" :
    
    # Load stock data
    ticker = 'PSA' # There as a small difference between Yahoo and mine calc, due to rounding, accumulated to about 2.4% over 30y
    
    test_with_actual_data = True
    
    if test_with_actual_data:
        # Load Datafile
        dfs_dir = "../stock_dfs"
        fpath = os.path.join(dfs_dir, f'{ticker}.csv')
        df = pd.read_csv(fpath, index_col = "Date", parse_dates=True)
    
    else: 
        # Build a simplified example
        df = build_test_file()
        
    # Test the function with dense inputs for dividends and splits
    closes    = df['Close']
    dividends = df['Dividends']
    
    if not test_with_actual_data:
        #splits = df['Stock Splits']
        splits = None
    else:
        splits = None
    
    adj_closes1 = calc_adj_closes(closes, dividends=dividends, splits=splits)
                         
    print(adj_closes1.head())
    
    # Test the function with sparse inputs for dividends and splits
    if dividends is not None:
        div_sparse    = dividends[dividends > 0].dropna()
    else:
        div_sparse    = None
        
    if splits is not None:
        splits_sparse = splits[splits > 0].dropna()
    else:
        splits_sparse    = None

    adj_closes2 = calc_adj_closes(closes, dividends=div_sparse, splits=splits_sparse)
                         
    print(adj_closes2.head())
    
    # Check if dense and sparse are equal
    check = adj_closes1['Close'].equals(adj_closes2['Close'])
    
    print(f"Dense and Sparse are equal: {check}")
