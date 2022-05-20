# -*- coding: utf-8 -*-
"""
Created on Thu May 19 07:32:13 2022
Analyse simulation postions vs. market cap
Convert positions into market cap, and check whether I was net more long or short
larger cap stocks
@author: vragu
"""
import pandas as pd
import os 
import yfinance as yf
import numpy as np
import concurrent.futures
import pickle

DATA_DIR = "../data"
STOCK_DFS_DIR  = "../stock_dfs"

#%% Analyse positions, identify most consistent longs and shorts
def an_longs_shorts():
    
    positions = pd.read_pickle(os.path.join(DATA_DIR,"positions.p"))

    # Calculate % long/short
    pos_count = positions.apply(pd.value_counts).T
    pos_count['total'] = pos_count.sum(axis=1)
    
    prc_long = pos_count[1.0]/pos_count['total']
    prc_shrt = pos_count[-1.0]/pos_count['total']
    
    #Sort in descending order
    prc_long = prc_long.sort_values(ascending=False)
    prc_shrt = prc_shrt.sort_values(ascending=False)
    
    print("Top Longs:", prc_long[:30],"\n")
    print("Top Shorts:", prc_shrt[:30])

    return prc_long, prc_shrt

#%% Load undajusted close-to-close returns for all stocks
def load_field_all_stocks(tickers, field='Close', reload = True, fname=None, save=False):
    """ Load data for a specified field for all stocks, returns a dataframe
    """
    
    # If a pickle with data is specified, load and return it
    if not reload:
        return pd.read_pickle(fname)

    # Otherwise, loop over csv files and construct the returns series
    df_all = pd.DataFrame()
    
    for count, ticker in enumerate(tickers):
        print(f"Processing {count}: {ticker}")
        df = pd.read_csv('{}/{}.csv'.format(STOCK_DFS_DIR, ticker))
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
       
        # Drop missing rows
        old_rows = len(df)
        df = df.dropna(subset=[field])
        if len(df) < old_rows:
            print(f"    Dropped {old_rows - len(df)} rows")
       
        # Convert into long format table so it's easier to aggregate across stocks
        df1 = df[[field]].reset_index()
        df1.columns = ['Date', field]
        df1['Ticker'] = ticker
        
        # Rearrange columns to make it easier to read
        df1 = df1[['Ticker', 'Date', field]]
        
        # Append the ticker to the large data frame
        if df_all.empty:
            df_all = df1
        else:
            df_all = pd.concat([df_all, df1]).reset_index(drop=True)
        
    print(f"Finished. Aggregate df has {len(df_all)} data rows.")

    # Flatten the data frame to have tickers in columns
    df_out = df_all.pivot_table(values=field, index='Date',columns=['Ticker'])    
    
    if save:
        df_out.to_pickle(fname)
        
    return df_out
    
#%% Correlate longs and shorts with market cap
def an_mkt_cap():
    
    # Load file with end-of-period market caps
    f_mc = 'mcap_updated.csv'
    mcap_last = pd.read_csv(os.path.join(DATA_DIR,f_mc),index_col="ticker", thousands=',')
    
    # Generate a market cap proxy for every month-end
    # Scale over time by unadjusted closes (not adjusted closes)
    # Unadjusted closes incorporate splits, but not dividends
    reload=False # Generate data from individual stock files if True/else load from picle
    tickers = mcap_last.index
    f_cls = os.path.join(DATA_DIR, "all_closes.p")
    closes = load_field_all_stocks(tickers=tickers,field='Close',fname=f_cls, reload=reload, save=reload)
    
    # Do a loop for now, later can re-write in a Pythonic way
    df_mc = closes.copy()
    for ticker in tickers:
        df_mc.loc[:,ticker] = df_mc.loc[:,ticker] * mcap_last.loc[ticker,"shrs_out"]
          
    # =============================================================================
    #     # Calc average market cap of longs and shorts
    # =============================================================================
    positions = pd.read_pickle(os.path.join(DATA_DIR,"positions.p"))
    
    # Get market caps on position dates
    mc = df_mc.resample('M').last()
    
    # We probably don't need it, but just in case, align mc with positions
    pos1, mc1 = positions.align(mc,join="left", axis=0)

    # Drop duplicate listings to avoid the bias
    # tickers_drop = ["GOOG"]
    # pos1 = pos1.drop(tickers_drop, axis=1)
    # mc1  = mc1.drop( tickers_drop, axis=1)

    # Filter market cap dataframe for longs and shorts
    longs  = np.where(pos1 > 0, mc1, np.nan)
    shorts = np.where(pos1 < 0, mc1, np.nan)

    mc_avg = pd.DataFrame(index=pos1.index, columns=['mcap_avg_longs','mcap_avg_shorts'])
    mc_avg['mcap_avg_longs']  = np.nanmean(longs,  axis=1) / (10**9)
    mc_avg['mcap_avg_shorts'] = np.nanmean(shorts, axis=1) / (10**9)
                            
    # Calculate top longs and shorts - exclude days when we have no positions
    top_longs_idx  = (-longs ).argsort(axis=1)[:,:3]
    top_shorts_idx = (-shorts).argsort(axis=1)[:,:3]
    
    top_longs  = np.apply_along_axis(lambda row: pos1.columns[row], 0, top_longs_idx )
    top_shorts = np.apply_along_axis(lambda row: pos1.columns[row], 0, top_shorts_idx)
    
    mc_avg['top_longs']  = top_longs [:,0] + " " + top_longs [:,1] + " " + top_longs [:,2]
    mc_avg['top_shorts'] = top_shorts[:,0] + " " + top_shorts[:,1] + " " + top_shorts[:,2]
    
    # Clean out data at the start before we have started trading
    mc_avg = mc_avg.dropna(subset=['mcap_avg_longs','mcap_avg_shorts'])

    # Make pretty graphs
    ax = mc_avg[['mcap_avg_longs','mcap_avg_shorts']].plot(title='Average Market Cap of Longs and Shorts')
    ax.set_ylabel('Average Market Cap, $bb', fontsize="small")
    ax.set_xlabel("Date",fontsize="small")
    ax.legend(["Longs","Shorts"])
          
    # Return data
    return mc_avg, mc
    
#%% Start of Main
if __name__ == "__main__":
    
    # Analyze positions - top longs/shorts
    prc_long, prc_shrt = an_longs_shorts()
    
    
    # Look at longs/short vs. their market cap
    mc_avg, mc = an_mkt_cap()
    
    # Generate a market cap proxy for every month-end
    # Calc average market cap of longs and shorts
    # Make pretty graphs
    print("\nFinished")
