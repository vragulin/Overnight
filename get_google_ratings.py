# -*- coding: utf-8 -*-
"""
Created on Fri May 20 23:01:39 2022
Calculate popularity ratings for stocks
@author: vragu
"""
import pytrends
from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import os
#import pickle

DATA_DIR = "../data"

#%% Get interest for a list of tickers
def get_goog_tickers(tickers):
    """ works for n<=5 or so"""
    
    #Create a list of keywords
    keywords = []
    for ticker in tickers:
        keywords.append(" ".join([ticker," share price"]))

    # Send keywords to Google
    trending_terms = TrendReq(hl='en-US', tz=360)
    trending_terms.build_payload(
          kw_list=keywords,
          cat=0,
          timeframe='today 5-y',
          geo='US',
          gprop='')
    term_interest_over_time = trending_terms.interest_over_time()
    
    # Calculate average interest
    goog_avg   = term_interest_over_time.mean()
    
    # Drop last row and add a column of tickers
    goog_avg1 = goog_avg.iloc[:-1,]
    
    df_goog = pd.DataFrame(goog_avg1.values, index=tickers, columns = ["avg_interest"])
    
    return df_goog

#%% Get interest for a single ticker
def get_goog_ticker(ticker):
    
    #Create a list of keywords
    keywords = [ " ".join([ticker,"share price"])]
                
    # Send keywords to Google
    trending_terms = TrendReq(hl='en-US', tz=360)
    trending_terms.build_payload(
          kw_list=keywords,
          cat=0,
          timeframe='today 5-y',
          geo='US',
          gprop='')
    term_interest_over_time = trending_terms.interest_over_time()
    
    # Calculate average interest
    if len(term_interest_over_time) == 0:
        return 0
    else:
        goog_avg   = term_interest_over_time.mean()       
        return goog_avg[0]


#%% Start of  Main
# =============================================================================
if __name__ == "__main__":

    tickers = pd.read_pickle(os.path.join(DATA_DIR,"sp500tickers.pickle"))
    
    goog = pd.DataFrame(np.nan, index=tickers, columns=["avg_interest"])
    
    for ticker in tickers:
        interest = get_goog_ticker(ticker)
        print(f"{ticker}: {interest}")
        goog.loc[ticker,"avg_interest"] = interest

    print("Done")