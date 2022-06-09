# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 21:55:48 2022

Test validate_one_stock function
@author: vragu
"""
import select_good_stocks as sgs

if __name__ == "__main__":

    # Load context (necessary params and data)
    context = sgs.load_context()

    # Run analysis for one ticker    
    ticker = "ZM"
    
    df, stats = sgs.validate_one_stock(ticker, context)
    
    print(df)
    print("\n")
    print(stats)
    