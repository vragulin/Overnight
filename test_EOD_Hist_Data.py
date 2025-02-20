# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 09:58:54 2022
Test EOD_Hist_Data
@author: vragu

EOD python api from: https://github.com/femtotrader/python-eodhistoricaldata/blob/master/eod_historical_data/data.py

"""


import pandas as pd
import eod_historical_data as eod
from eod_historical_data._utils import (_init_session, _format_date,
                     _sanitize_dates, _url, RemoteDataError)
import datetime
import requests
import requests_cache
from io import StringIO

# My config file with API Keys
import config

EOD_HISTORICAL_DATA_API_KEY_DEFAULT = config.EOD_API_KEY
EOD_HISTORICAL_DATA_API_URL = "https://eodhistoricaldata.com/api"

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
        # df = pd.read_csv(StringIO(r.text), skipfooter=1,  <- don't need to skip footer
        df = pd.read_csv(StringIO(r.text), 
                         parse_dates=[0], index_col=0, engine='python')
        assert len(df.columns) == 1
        ts = df["Stock Splits"]
        return ts
    else:
        params["api_token"] = "YOUR_HIDDEN_API"
        raise RemoteDataError(r.status_code, r.reason, _url(url, params))
#%% Entry point
if __name__ == "__main__":
    
    
    # Cache session (to avoid too much data consumption)
    expire_after = datetime.timedelta(days=1)
    session = requests_cache.CachedSession(cache_name='cache', backend='sqlite',
                                           expire_after=expire_after)
    
    sym = "AAPL"
    exchange = "US"
    start = "1993-01-01"
    end = "2022-06-01"
    
    #Historical prices
    df = eod.get_eod_data(sym, exchange, start=start, end=end, session=session)
    print(df.tail())
    
    #Dividends
    df_div = eod.get_dividends(sym, exchange, start=start, end=end, session=session)
    print(df_div)
    
    #Splits
    df_splits = get_splits(sym, exchange, start=start, end=end, session=session)
    print(df_splits)
    