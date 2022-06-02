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
import yfinance as yf
#import pickle

DATA_DIR = "../data"

#%% Get interest for a list of tickers
def get_goog_tickers(tickers):
    """ works for n<=5 or so"""
    
    #Create a list of keywords
    keywords = []
    for ticker in tickers:
        tkObj = yf.Ticker(ticker)
        name=tkObj.info['shortName']
        print(f"{ticker}: {name}")
        keywords.append(" ".join([name," share price"]))

    # Send keywords to Google
    trending_terms = TrendReq(hl='en-US', tz=360)
    trending_terms.build_payload(
          kw_list=keywords,
          cat=784,
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
    tkObj = yf.Ticker(ticker)
    #name=tkObj.info['shortName']
    name = ticker
    print(f"{ticker}: {name}")
    #keywords = [ " ".join([name[:12],"shares"])]
    # if name in ["Amazon.com", "Netflix, Inc."]:
    #     keywords = [ "Amazon.com", "Netflix, Inc." ]
    # else:
    #     keywords = [ "Amazon.com", "Netflix, Inc.", name ]
    
    ref_keywords = ['AMZN', 'NFLX']
    
    if ticker not in ref_keywords:
        ref_keywords.append(name)
    
    keywords = ref_keywords
    
    suffix  = " stock"
    adj_keywords = [ kw + suffix for kw in keywords ]
    
    # Send keywords to Google
    trending_terms = TrendReq(hl='en-US', tz=360)
    trending_terms.build_payload(
          kw_list=adj_keywords,
          cat=784,  #Business News
          timeframe='today 5-y',
          geo='US',
          gprop='')
    term_interest_over_time = trending_terms.interest_over_time()
    
    # Calculate average interest
    if len(term_interest_over_time) == 0:
        return 0
    else:
        goog_avg   = term_interest_over_time[name+suffix].mean()       
        return goog_avg


#%% Start of  Main
# =============================================================================
if __name__ == "__main__":

    tickers = pd.read_pickle(os.path.join(DATA_DIR,"sp500tickers.pickle"))
    
    goog = pd.DataFrame(np.nan, index=tickers, columns=["avg_interest"])
    
    for ticker in tickers: #['AMZN', 'NFLX', 'TECH', 'SO', 'MMM']:
        interest = get_goog_ticker(ticker)
        print(f"{ticker}: {interest}")
        goog.loc[ticker,"avg_interest"] = interest
    
    goog.to_clipboard()
    print("Done")
    