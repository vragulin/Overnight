# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 21:46:06 2022
Build an index of sim stocks
Compare vs. SPY or other benchmarks
Plan:
    * Load sim stocks into a dataframa
    * Get all adjusted closes and market caps into dataframes
    * Calculate weights and returns
    * Calc Index
    * Compare
@author: vragu
"""
import pandas as pd
import numpy as np
import pickle, os
import config as cf

#%% Global variables - file locations
MONTHLY_FIELD_DF = "sim_monthly_{field}.pickle"

#%% Load ticker list and stock panel data 
def load_data():
    
    # Load tickers
    with open(os.path.join(cf.DATA_DIR, cf.TICKER_FILE), "rb") as f:
        tickers = pickle.load(f)

    # Load stock panel dataframe
    df_all_stocks = pd.read_pickle(os.path.join(cf.DATA_DIR, cf.ALL_STOCKS_DFS))
    
    return tickers, df_all_stocks

#%% Build a monthly dataframe of overnight returns
def extract_field_for_all_tickers(field, tickers, df_all_stocks, save_file=None):
    """ Extract a dataframe with rows=dates, cols = stocks, values = field from 
        the large database
        
        Parameters:
            field       - name of a field that we want to load for every stock
            tickers     - list of tickers
            df_all_stocks - database with all fields / dates / stocks
            save_file   - save the result into a pickle file, if None - don't save
            
        Return - data dataframe
    """
    
    fnames = {"Adj Close" : "Adj_Close", "Mkt Cap": "Mkt_Cap"}
    pickle_fname = MONTHLY_FIELD_DF.format(field=fnames[field])

    
    # Re-generate monthy on returns form panel data
    df1 = df_all_stocks.pivot_table(values='Value', index='Date', columns=['Ticker', 'Field'])
        
    # Extract Intraday Returns for all stocks
    df2 = df1.swaplevel(axis=1)[field]
    
    # Resample to monthly frequency
    dfm = df2.resample('M').fillna("ffill")

    # Save new df into a pickle
    if save_file:
        dfm.to_pickle(os.path.join(cf.DATA_DIR, pickle_fname))
            
    return dfm

#%% Calculate weights
def calc_mcap_weights(mcap, drop_missing=True, equal_weight=False):
    """ From a dataframe of market caps calculate index weights
        Parameters:
            - mcap : pandas datarfame of market caps for each stock
            - drop_missing : if True, then if for a stock i, mcap(i,t+1) = nan, set w(i,t)=0
                             otherwise, w(i,t) = mcap(i,t) / sum(mcap(:,t))
            - equal_weight : if equal-weighted index
    """
    
    
    # if drop_missing=True, set mcap(i,t) to np.nan if mcap(i,t+1))=np.nan
    if drop_missing:
        mcap1 = np.where(mcap.shift(-1).isnull(), np.nan, mcap)
    else:
        mcap1 = mcap.values
        
    # if the user specified and equal-weighted index, replace actual mkt caps with ones
    if equal_weight:
        mcap1 = (~np.isnan(mcap1)).astype(float)
        
    # Caclucate the sum of all valid market caps for each day
    idx_cap = np.nansum(mcap1, axis=1)

    # Scale down all market caps by the index cap    
    #weights = mcap1 / idx_cap[:,None]
    weights = mcap1 / idx_cap.reshape(-1,1)
    
    df = pd.DataFrame(weights, columns=mcap.columns, index=mcap.index)

    return df

#%% Calculate index return
def calc_index_returns(adj_close, weights):
    """ Calculate index returns 
        Parameters:
            - adj_close: pandas dataframe of adjusted_close prices (or cum_returns)
            - weights : weight(t,i). 
            
        Return:
            - df : pandas dataframe of index cumulative and period returns
                   r_idx(t,i) = sum[ w(t-1,i) * p(t,i)/p(t-1,i)] - 1
    """
    
    simple_rets = adj_close.pct_change()
    weighted_rets = simple_rets * weights.shift(1)
    idx_rets = np.nansum(weighted_rets.values, axis=1)
    cum_rets = (1+idx_rets).cumprod()
    
    df = pd.DataFrame(np.column_stack((idx_rets, cum_rets)),
                      columns=['simple_rets', 'cum_rets'], index=adj_close.index)
    
    return df
    
#%% Main Program
def main():
   
   
    # =============================================================================
    # Load dataframes for Adj Close and Market Cap
    # =============================================================================
    # Build path files for Adj_Close and MCap dataframes
    adj_close_f = os.path.join(cf.DATA_DIR, MONTHLY_FIELD_DF.format(field="Adj_Close"))
    mcap_f      = os.path.join(cf.DATA_DIR, MONTHLY_FIELD_DF.format(field="Mkt_Cap"  ))
    
    # If dataframes have already been saved, load them, otherwise - build from the big database
    rebuild_frames = False
    if rebuild_frames:      # Extract the needed fields from the big database
        # Load data
        tickers, df_all_stocks = load_data()
        
        # Extract Adj Closes
        adj_close = extract_field_for_all_tickers("Adj Close", tickers, df_all_stocks,
                                                  save_file = adj_close_f)

        # Extract Mkt Caps        
        mcap      = extract_field_for_all_tickers("Mkt Cap",   tickers, df_all_stocks,
                                                  save_file = mcap_f)
        
        # Extract is liquidity conditions are satistifed on a given day
    
    else:               # Already have the dataframes saved, just load them
        # Load tickers
        with open(os.path.join(cf.DATA_DIR, cf.TICKER_FILE), "rb") as f:
            tickers = pickle.load(f)
        
        # Load Adj Close and and Mcap from pickle 
        adj_close =  pd.read_pickle(adj_close_f)
        mcap      =  pd.read_pickle(mcap_f)

    # =============================================================================
    # Calculate index
    # =============================================================================
    # Calculate weights
    weights = calc_mcap_weights(mcap, equal_weight = True)
    check = weights.sum(axis=1)
    print("Checking that weights add up to 1", check[:5])
    
    df_rets = calc_index_returns(adj_close, weights)
    df_rets.cum_rets.plot()
   
    # Save portfolio returns to a 
    df_rets.to_pickle(os.path.join(cf.DATA_DIR,"sim_port_rets.pickle"))
   
    return df_rets

#%% Entry point
if __name__ == "__main__":
    df_rets = main()
    