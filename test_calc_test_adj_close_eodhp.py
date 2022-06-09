# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:51:45 2022

Test program for calc_adj_close_eodhp
@author: vragu
"""

import pandas as pd
import os

import calc_adj_close as cac


#%% Entry Point
if __name__ == "__main__":
    
    ticker = "TJX"
    
    # =============================================================================
    # Load stock data: price, splits and dividends
    # =============================================================================
    price_file = os.path.join("../stock_dfs", f"{ticker}.csv")
    df_px = pd.read_csv(price_file, index_col = 'Date', parse_dates = True)

    div_file = os.path.join("../div_dfs", f"{ticker}.csv")
    df_div = pd.read_csv(div_file, index_col = 'Date', parse_dates = True)

    split_file = os.path.join("../split_dfs", f"{ticker}.csv")
    df_split = pd.read_csv(split_file, index_col = 'Date', parse_dates = True)

    # =============================================================================
    # Convert splits into the default format
    # =============================================================================
    df_split1 = cac.parse_eod_hd_splits(df_split['Stock Splits'])
    print(df_split1)

    # =============================================================================
    # Test calc_adj_closes functions with default split format
    # =============================================================================
    df_adj = cac.calc_adj_closes(df_px['Close'],splits=df_split1['Split Factor'])
    print(df_adj.head())
    print(df_adj.tail())

    # =============================================================================
    # Test calc_adj_closes functions with eod_hd split format
    # =============================================================================
    df_adj = cac.calc_adj_closes(df_px['Close'],splits=df_split['Stock Splits'], source='eod_hd')
    print(df_adj.head())
    print(df_adj.tail())
       
    
    # =============================================================================
    # Test adj close vs. Yahoo, now also add a dividend adjustment
    # =============================================================================
    df_adj = cac.calc_adj_closes(df_px['Close'],dividends = df_div['Dividends'], splits=df_split['Stock Splits'], source='eod_hd')
    print(df_adj.head())
    print(df_adj.tail())

    print("Done")