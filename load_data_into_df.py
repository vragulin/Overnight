# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:03:16 2021

@author: vragu
"""

import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
import numpy as np

# Global variables - file locations
TICKER_FILE    = "data/sp500tickers.pickle"
ALL_STOCKS_DFS = "data/all_stocks_px_ret.pickle"
STOCK_DFS_DIR  = "stock_dfs"

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
def get_data_from_yahoo(reload_sp500=False):
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
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            #df = df.drop("Symbol", axis=1)
            df.to_csv('{}/{}.csv'.format(STOCK_DFS_DIR, ticker))
        else:
            print('Already have {}'.format(ticker))



#%% Collect data into a single dataframe
def compile_data():
    
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        print(f"Processing {count}: {ticker}")
        df = pd.read_csv('{}/{}.csv'.format(STOCK_DFS_DIR, ticker))
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
       
        # Calculte adjusted open and returns
        df['Factor'] = df['Close'] / df['Adj Close']
        df['Adj Open'] = df ['Open'] / df['Factor']

        # Drop missing rows
        old_rows = len(df)
        df = df.dropna(subset=['Adj Close', 'Adj Open'])
        if len(df) < old_rows:
            print(f"    Dropped {old_rows - len(df)} rows")
        
        # Calculate log returns
        df['r_full'] = np.log(df['Adj Close']).diff()
        df['r_intr'] = np.log(df['Adj Close']) - np.log(df['Adj Open'])
        df['r_ovnt'] = df['r_full'] - df['r_intr']

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
    #get_data_from_yahoo(reload_sp500=True)
    compile_data()


if __name__ == "__main__":
    #main()
    compile_data()
