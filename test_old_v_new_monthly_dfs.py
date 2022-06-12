# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 19:41:42 2022

Compare old vs. new dataframes
@author: vragu
"""

#import pickle
import pandas as pd


if __name__ == "__main__":

    field = 'r_ovnt'
    ticker = 'XOM'
    
    df_old = pd.read_pickle(f'../data/monthly_{field}.pickle')
    df_new = pd.read_pickle(f'../data/elm_monthly_{field}.pickle')
        
    print("Old data:")
    print(df_old[ticker].tail())

    print("\nNew data:")
    print(df_new[ticker].tail())

    print('Done')
    

