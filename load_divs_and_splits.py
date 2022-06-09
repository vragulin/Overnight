# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 07:28:10 2022
Load dividends and splits from EOD Historical Data
@author: vragu
"""

import pandas as pd
import eod_historical_data as eod
from eod_historical_data._utils import (_init_session, _format_date,
                     _sanitize_dates, _url, RemoteDataError)
import datetime
import requests
import requests_cache
from io import StringIO
import os
import load_data_into_df as ld

# My config file with API Keys
import config

#%% Global Variables
EOD_HISTORICAL_DATA_API_KEY_DEFAULT = config.EOD_API_KEY
EOD_HISTORICAL_DATA_API_URL = "https://eodhistoricaldata.com/api"

DIV_DFS_DIR     = '../div_dfs'
SPLIT_DFS_DIR   = '../split_dfs'

#%% Get splits
def get_splits(symbol, exchange, start=None, end=None,
                  api_key=EOD_HISTORICAL_DATA_API_KEY_DEFAULT,
                  session=None):
    """
    Returns splits
    """
    symbol_exchange = symbol + "." + exchange
    session = _init_session(session)
    start, end = _sanitize_dates(start, end)
    endpoint = "/splits/{symbol_exchange}".format(symbol_exchange=symbol_exchange)
    url = EOD_HISTORICAL_DATA_API_URL + endpoint
    params = {
        "api_token": api_key,
        "from": _format_date(start),
        "to": _format_date(end)
    }
    r = session.get(url, params=params)
    if r.status_code == requests.codes.ok:
        df = pd.read_csv(StringIO(r.text), skipfooter=1,
                         parse_dates=[0], index_col=0, engine='python')
        assert len(df.columns) == 1
        ts = df["Stock Splits"]
        return ts
    else:
        params["api_token"] = "YOUR_HIDDEN_API"
        raise RemoteDataError(r.status_code, r.reason, _url(url, params))

#%% Get splits nad dividends and save into the data directories
def get_divs_and_tickers_from_eodhist(tickers, start=None, end=None):
    """ Load Market Cap history from eodhist API for all tickers on the list.
        Save data into files in the respective directories """
   
    # Cache session (to avoid too much data consumption)
    expire_after = datetime.timedelta(days=1)
    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite',
                                           expire_after=expire_after)
    
    exchange = "US"
    
            
    # Loop over tickers and data fields
    fields = ["divs", "splits"]
    funcs  = [ eod.get_dividends, get_splits ]
    dirs   = [DIV_DFS_DIR, SPLIT_DFS_DIR ]

    # Ensure that the directories for data files exists
    for dfs_dir in dirs:
        if not os.path.exists(dfs_dir):
                os.makedirs(dfs_dir)
    
    for i, ticker in enumerate(tickers):

        for field, func, dpath in zip(fields, funcs, dirs):
        
            # just in case your connection breaks, we'd like to save our progress!
            if not os.path.exists('{}/{}.csv'.format(dpath, ticker)):
        
                try:
                    df = func(ticker, exchange=exchange, start=start, end=end, session=session)
                except RemoteDataError as e:
                    print(f"{i}: {ticker} - API call failed", e)
                    df = None
        
                if df is not None:            
                    if len(df) > 0:
                        df.to_csv('{}/{}.csv'.format(dpath, ticker))
                        print(f"{i}: {ticker} - {field} loaded")
                        continue
                
                # Print out if there is not data
                print(f"{i}: {ticker} - no {field} data")
            
            else:
                print(f"{i}: {ticker} - already have {field} data")
          


#%% Entry Point
if __name__ == "__main__":
    
        
    #   Get tickers
    tickers = ld.load_BBG_tickers(reload=False)

    # get divs and for all tickers
    start = "1993-01-01"
    end   = "2022-06-01"

    get_divs_and_tickers_from_eodhist(tickers)
    #get_divs_and_tickers_from_eodhist(["AANI"])
    
    
    # From number of shares and share prcies generate series of market cap proxies
    print("Done")

