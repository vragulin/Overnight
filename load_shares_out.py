# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 19:18:24 2022
Load market cap data
@author: vragu

"""

import datetime as dt
import os
import pandas as pd
#from eod_historical_data._utils import RemoteDataError
import load_data_into_df as ld
import json
import requests
import requests_cache
import config

SHRS_DFS_DIR = "../shrs_dfs"
START_DATE    = dt.datetime(1993,1,1)

#%% Get number of shares outstanding
def get_shrsOut_eodhist_one_stock(ticker, exchange="US", start=None, end=None, session=None):
    # Fetch market cap data for a single ticker
    api_key = config.EOD_API_KEY
      
    if session is None:
        session = requests.Session()
        
    # Pull historical market cap data for a single stock
    url = f"https://eodhistoricaldata.com/api/fundamentals/{ticker}.US?filter=Financials::Balance_Sheet::yearly"
    
    params = {"api_token": api_key }
    r = session.get(url, params=params)
    
    if r.status_code == requests.codes.ok:
        data = json.loads(r.text)
        if data != "NA":
            df = pd.DataFrame(data).T
        else:
            return None
        
        df.index.name = "Date"
        # The Date column sometimes has bad dates.  Get rid of them if necessary
        df.index = pd.to_datetime(df.index, errors='coerce')
        if any(df.index.isnull()):
            df = df[~df.index.isnull()]
  
        #Extract a correct date range.  Since we get annual data, we should get report at least 1 year ahead
        start_rpt = start - dt.timedelta(days=365)
        df_range = df[(df.index >= start_rpt) & (df.index <= end)]
        if len(df_range) == 0:
            return None
        try:
            return df_range[['commonStockSharesOutstanding']]
        except KeyError as e:
            print(f"Could not get data for {ticker}. ", e)
            return None
    
    else: 
        print(f"API failed for {ticker}, r.status_code={r.status_code}, r.reason={r.reason}")
        return None

# Get market cap data for a list of tickets
def get_shrsOut_from_eodhist(tickers):
    """ Load Market Cap history from eodhist API for all tickers on the list.
        Save data into files in the MCAP_DFS_DIR directory """
    
    start = START_DATE
    end = dt.datetime.now()

    expire_after = dt.timedelta(days=3)
    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite',
                                           expire_after=expire_after)
    exchange = "US"
    
    # Ensure that the directory for data files exists
    if not os.path.exists(SHRS_DFS_DIR):
        os.makedirs(SHRS_DFS_DIR)
        
    # Loop over tickers
    for i, ticker in enumerate(tickers):
        print(f"{i}: {ticker}")
        
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('{}/{}.csv'.format(SHRS_DFS_DIR, ticker)):

            df_shrs = get_shrsOut_eodhist_one_stock(ticker, exchange=exchange, 
                                            start=start, end=end, session=session)

            if df_shrs is not None:            
                # Rename columns to make consistent with yahoo
                df_shrs.rename(columns={'commonStockSharesOutstanding':'shrsOut'}, inplace = True)    
                df_shrs.to_csv('{}/{}.csv'.format(SHRS_DFS_DIR, ticker))
            else:
                print(f"Could not get data for {ticker}.")
        else:
            print('Already have {}'.format(ticker))
            
#%% Entry Point
if __name__ == "__main__":
    
        
    #   Get tickers
    tickers = ld.load_BBG_tickers(reload=False)

    # get market cap data for all tickers
    get_shrsOut_from_eodhist(tickers)
    #get_shrsOut_from_eodhist(["SVVC"])

    # From number of shares and share prcies generate series of market cap proxies
    print("Done")
