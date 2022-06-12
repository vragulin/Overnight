# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 19:41:42 2022

Compare old vs. new dataframes
@author: vragu
"""

#import pickle
import pandas as pd


if __name__ == "__main__":
    
    df_old = pd.read_pickle('../data/all_stocks_px_ret.pickle')
    
    df_new = pd.read_pickle('../data/elm_stocks_px_mcap.pickle')
    
    lag = 32
    ticker = 'XOM'
    
    tkr_old = df_old[df_old['Ticker'] == ticker]
    tkr_new = df_new[df_new['Ticker'] == ticker]
    
    ref_date = tkr_new.iloc[-lag]['Date']
    
    tkr_old_ref = tkr_old[tkr_old['Date']==ref_date]
    tkr_new_ref = tkr_new[tkr_new['Date']==ref_date]
    
    print("Old data:")
    print(tkr_old_ref)

    print("\nNew data:")
    print(tkr_new_ref)

    print('Done')
    

