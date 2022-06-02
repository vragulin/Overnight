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
def calc_adj_closes(closes, dividends=None, splits=None):
    """
    Parameters
    ----------
    close : pandas series with daily frequency (later may be able to adapt for other frequencies)
    dividents : panda series with dividend amounts and dates, can be either dense (for all dates)
                or sparse (without zeros), if None - don't adjust
    splits : pandas series with splits dates and rates, dense or sparse, if None - don't adjust
    
    Returns
    -------
    df: dataframe with adjusted closes prices for same dates as closes, splits factors and div factors
    """

    # Create a copy of dividends
    adj_dividends = dividends.copy()
    adj_dividends.name = "Dividends"
    
    # Adjust for splits
    df = pd.DataFrame(closes).rename(columns={closes.name: "Close"})
    
    
    if splits is not None:
        df = df.join(splits[splits>0], how='outer').sort_index(ascending=False)\
                    .rename(columns={splits.name: "Splits"})
        split_coef = df['Splits'].shift(1).fillna(1).cumprod()
        df['Close'] = df['Close'] / split_coef
        
        if dividends is not None:
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
