# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:03:16 2021

@author: vragu
"""
#%% Load modules
import bs4 as bs
import datetime as dt
import os, sys
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import pickle
import requests
import numpy as np
from calc_adj_close import calc_adj_closes
import config
from alpha_vantage.timeseries import TimeSeries

# Global variables - file locations
TICKER_FILE    = "../data/sp500tickers.pickle"
ALL_STOCKS_DFS = "../data/all_stocks_px_ret.pickle"
STOCK_DFS_DIR  = "../stock_dfs"

#%% Load S&P Tickers
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.rstrip().replace('.','-'))
    with open(TICKER_FILE, "wb") as f:
        pickle.dump(tickers, f)
    return tickers


#%% Load historical prices into CSV files
def get_data_from_yahoo(reload_sp500=False, source='web_yahoo'):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open(TICKER_FILE, "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists(STOCK_DFS_DIR):
        os.makedirs(STOCK_DFS_DIR)

    start = dt.datetime(1993, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        print(ticker)
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('{}/{}.csv'.format(STOCK_DFS_DIR, ticker)):
            
            # Choose source for the data
            if source == 'web_yahoo':            
                df = web.DataReader(ticker, 'yahoo', start, end)
            elif source == 'yfinance':
                tk = yf.Ticker(ticker)
                df = tk.history(start=start, end=end, auto_adjust=False)
            elif source == 'alpha-vantage':
                df = web.DataReader(ticker, 'av-daily-adjusted',start=start, end=end, \
                                    api_key=config.AV_API_KEY)
                # Rename columns to make consistent with yahoo
                df.index.name = "Date"
                df.rename(columns={'open':'Open', 'high':'High','low':'Low','close':'Close',\
                                   'adjusted close':'Adj Close','volume':'Volume',\
                                   'dividend amount':'Dividends','split coefficient':'Stock Splits'},\
                          inplace = True)
            else:
                raise NotImplementedError(f"Source {source} not implemented")
            
            # Format and save data in the right way
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            #df = df.drop("Symbol", axis=1)
            df.to_csv('{}/{}.csv'.format(STOCK_DFS_DIR, ticker))
        else:
            print('Already have {}'.format(ticker))



#%% Collect data into a single dataframe
def test_my_adj_close(df):
    """ Test that my adj close is similar to yahoo, if not print message"""
    tolerance = 0.001 
    if np.abs(df['Adj Close'].iloc[0] / df['Adj Close old'].iloc[0]-1) > tolerance:
        print("My adj close does not match data source\n")
    return None

def compile_data(use_own_adj = False, source='web_yahoo', drop_splits=False):
    """ Build a data frame from csv files """
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers): #enumerate(['DD']): 
        print(f"Processing {count}: {ticker}")
        df = pd.read_csv('{}/{}.csv'.format(STOCK_DFS_DIR, ticker))
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
       
        # if requested, calculate my own adjusted close
        if use_own_adj:
            if source == 'alpha-vantage':
                print(f"Source={source} not implemented for div/split adj, using data from source.")
            else:    
                df_adj = calc_adj_closes(df['Close'],dividends=df['Dividends'],splits=None)
                df.rename(columns={'Adj Close':'Adj Close old'},inplace=True)
                df['Adj Close'] = df_adj['Close']
                _  = test_my_adj_close(df)


        # Calculte adjusted open and returns
        df['Factor'] = df['Close'] / df['Adj Close']
        df['Adj Open'] = df ['Open'] / df['Factor']

        # Drop missing rows - don't do it since it may imply that we hold position for several days
        # old_rows = len(df)
        # df = df.dropna(subset=['Adj Close', 'Adj Open'])
        # if len(df) < old_rows:
        #     print(f"    Dropped {old_rows - len(df)} rows")
        
        # Drop unneeded columns to save memory
        df = df.drop(columns = ['High','Low','Volume'])
                    
        # Calculate log returns
        df['r_full'] = np.log(df['Adj Close']).diff()
        df['r_intr'] = np.log(df['Adj Close']) - np.log(df['Adj Open'])
        df['r_ovnt'] = df['r_full'] - df['r_intr']

        # If requested, drop dates of splits since they have a high chance of errors due to spinoffs
        if drop_splits:
            df["r_ovnt"] = np.where((df["Stock Splits"] > 0) & (df["Stock Splits"] != 1), np.nan, df["r_ovnt"])
            df["r_full"] = np.where((df["Stock Splits"] > 0) & (df["Stock Splits"] != 1), np.nan, df["r_full"])
            
        # Convert into long format table so it's easier to aggregate across stocks
        df1 = df.unstack().reset_index()
        df1.columns = ['Field', 'Date', 'Value']
        df1['Ticker'] = ticker
        
        # Rearrange columns to make it easier to read
        df1 = df1[['Ticker', 'Date', 'Field','Value']]
        
        # Append the ticker to the large data frame
        if main_df.empty:
            main_df = df1
        else:
            main_df = pd.concat([main_df, df1]).reset_index(drop=True)
            
    print(f"Finished. Aggregate df has {len(main_df)} rows.")
    with open(ALL_STOCKS_DFS, "wb") as f:
        pickle.dump(main_df, f)
    
def main():
    
    #save_sp500_tickers()
    #get_data_from_yahoo(reload_sp500=True, source='yfinance')
    compile_data(use_own_adj=False, drop_splits=True)


if __name__ == "__main__":
    main()
    
