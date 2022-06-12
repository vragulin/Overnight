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
import matplotlib.pyplot as plt

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
def extract_field_for_all_tickers(field, tickers, df_all_stocks, save_file=None, 
                                  resample_func = 'last'):
    """ Extract a dataframe with rows=dates, cols = stocks, values = field from 
        the large database
        
        Parameters:
            field       - name of a field that we want to load for every stock
            tickers     - list of tickers
            df_all_stocks - database with all fields / dates / stocks
            save_file   - save the result into a pickle file, if None - don't save
            
        Return - data dataframe
    """
    
    fnames = {"Adj Close" : "Adj_Close", "Mkt Cap": "Mkt_Cap", "all_window": "all_window" }
    pickle_fname = MONTHLY_FIELD_DF.format(field=fnames[field])

    # Re-generate monthy on returns form panel data
    df1 = df_all_stocks.pivot(values='Value', index='Date', columns=['Ticker', 'Field'])
        
    # Extract Intraday Returns for all stocks
    df2 = df1.swaplevel(axis=1)[field]
    
    # Resample to monthly frequency
    if resample_func == 'sum':
        dfm = df2.resample('M').sum() 
    elif resample_func == "last":
        dfm = df2.resample('M').fillna("ffill") 
    else:
        print(f"Error: invali resample func : {resample_func}")
        return None

    # Save new df into a pickle
    if save_file:
        dfm.to_pickle(os.path.join(cf.DATA_DIR, pickle_fname))
            
    return dfm

#%% Build a monthy dataframe with index members (as of the start of the period)
def gen_index_members(eligible):
    """ Generate a historical dataframe of members of our index
    Parameters:
        - eligible: pandas dataframe with True/False boolean indicating whether a stock i is eligible for an index on date t
    
    Return : in_idx : boolean with index=dates, columns=stocks indciating whether a stock is actually in the index on the close
                        of this day
    """
    
    # Start with the simplest definition.  Stock goes into the index if it's eligible at the month-end (starting next month)
    #   and it exits the index (at month-end) as soon as it's no longer eligible.
    
    in_idx = eligible
    
    return in_idx
    
#%% Calculate weights
def calc_mcap_weights(mcap, idx_members=None, drop_missing=False, equal_weight=False):
    """ From a dataframe of market caps calculate index weights
        Parameters:
            - mcap : pandas datarfame of market caps for each stock
            - idx_members : pandas dataframe of booleans indicating whether a stock is an index member on the close
                            for the avoidance of doubt, if the stock goes into the index at M/E, the flag is TRUE for this day,
                            and if it is exits the index, the flag is FALSE
            - drop_missing : if True, then if for a stock i, mcap(i,t+1) = nan, set w(i,t)=0
                             otherwise, w(i,t) = mcap(i,t) / sum(mcap(:,t))
            - equal_weight : if equal-weighted index
    """
    
    
    # if drop_missing=True, set mcap(i,t) to np.nan if mcap(i,t+1))=np.nan
    # this is a dangerous flag, since it excludes stocks that went down, so potentially introduces
    # a look-ahead bias, try not to use it.
    
    if drop_missing:
        mcap1 = np.where(mcap.shift(-1).isnull(), np.nan, mcap)
    else:
        mcap1 = mcap.values.copy()
    
    # Only consider index members:
    if idx_members is not None:
        mcap1 = np.where(idx_members==False, np.nan, mcap1)
        
    # if the user specified and equal-weighted index, replace actual mkt caps with ones
    if equal_weight:
        mcap1 = (~pd.isna(mcap1)).astype(float)
        
    # Caclucate the sum of all valid market caps for each day
    idx_cap = np.nansum(mcap1, axis=1)

    # Scale down all market caps by the index cap    
    #weights = mcap1 / idx_cap[:,None]
    weights = mcap1 / idx_cap.reshape(-1,1)
    
    df = pd.DataFrame(weights, columns=mcap.columns, index=mcap.index)

    return df.astype(float)

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
    
#%% Analyse n largest weights
def n_largest_each_col(df, n=1):
    """ Get n largest elements from a series
    Parameters:
        - df : pandas dataframe
        - n : number of names to get
        - names : if True get indices rather than values

    Results:
        pandas dataframe.  First n columns with weights, then n columns with names

    """
    
    # Weights
    n_vals = (df.T).apply(lambda row : row.nlargest(n).values).T
    n_vals.columns = [f'w_{i+1}' for i in range(n)]
    
    # Tickers
    n_idx =  (df.T).apply(lambda row : row.nlargest(n).index).T
    n_idx.columns = [f'tkr_{i+1}' for i in range(n)]
    
    # Join together
    df = n_vals.join(n_idx, how='left')
    return df

#%% Main Program
def main():
   
    # =============================================================================
    # Load dataframes for Adj Close and Market Cap
    # =============================================================================
    # Build path files for Adj_Close and MCap dataframes
    adj_close_f = os.path.join(cf.DATA_DIR, MONTHLY_FIELD_DF.format(field="Adj_Close" ))
    mcap_f      = os.path.join(cf.DATA_DIR, MONTHLY_FIELD_DF.format(field="Mkt_Cap"   ))
    eligible_f  = os.path.join(cf.DATA_DIR, MONTHLY_FIELD_DF.format(field="all_window"))
    
    # If dataframes have already been saved, load them, otherwise - build from the big database
    rebuild_frames = False
    if rebuild_frames:      # Extract the needed fields from the big database
        # Load data
        tickers, df_all_stocks = load_data()
        
        # Extract Adj Closes
        adj_close = extract_field_for_all_tickers("Adj Close", tickers, df_all_stocks,
                                                  save_file = adj_close_f).astype(float)

        # Extract Mkt Caps        
        mcap      = extract_field_for_all_tickers("Mkt Cap",   tickers, df_all_stocks,
                                                  save_file = mcap_f).astype(float)
        
        # Extract is liquidity conditions are satistifed on a given day
        eligible  = extract_field_for_all_tickers("all_window",   tickers, df_all_stocks,
                                                  save_file = eligible_f).astype(float)
        
    else:               # Already have the dataframes saved, just load them
        # Load tickers
        with open(os.path.join(cf.DATA_DIR, cf.TICKER_FILE), "rb") as f:
            tickers = pickle.load(f)
            
        # Load Adj Close and and Mcap from pickle 
        adj_close =  pd.read_pickle(adj_close_f).astype(float)
        mcap      =  pd.read_pickle(mcap_f).astype(float)
        eligible  =  pd.read_pickle(eligible_f).astype(float)

    # Exclude names with bad data
    bad_data       = ['YELL','SPR', 'CBRL', 'HZNP', 'WTM', 'TEL', 'EMN']  #bad share count data or other obvious issues
    bad_corp_act   = ['KSU', 'SSP']    #incorrect representation of corporate actions
    free_float_adj = ['GOOG']
    eligible[bad_data + bad_corp_act + free_float_adj ] = False

    # Calculate index
    # =============================================================================
    # Build a dataframe of monthly of index members over time
    idx_members = gen_index_members(eligible)
    
    # Calculate weights
    weights = calc_mcap_weights(mcap, idx_members=idx_members, equal_weight = False)
    check = weights.sum(axis=1)
    print("Checking that weights add up to 1", check[:5])
    
    df_rets = calc_index_returns(adj_close, weights)
    df_rets.cum_rets.plot()
    plt.show()
   
    # Plot max weight and list names
    # df_w = pd.DataFrame(weights.max(axis=1,skipna=True),index=weights.index, columns=['max_weight'])
    # df_w['max_name'] = 0
    # for i in range(len(df_w)):
    #     w_row = weights.iloc[i,:]
    #     df_w.iloc[i,1] = w_row[w_row == df_w.iloc[i,0]].index[0]
        
    # df_w['max_weight'].plot()
    # plt.show()

    # As a check, look at the history of the names in the index
    df_largest = n_largest_each_col(weights,5)
    df_largest['w_1'].plot()
    plt.show()
    
    # As another check, look at stocks with the highest and lowest returns each months
    stk_rets = np.log(adj_close).diff()
    stk_rets = stk_rets * idx_members.shift(1)
    stk_rets = stk_rets.dropna(how='all').astype(float)
    
    ret_largest  = n_largest_each_col(stk_rets, 3)
    ret_smallest = n_largest_each_col(-stk_rets, 3)
    
    tests = [df_largest, ret_largest, ret_smallest ]
    # Save portfolio returns to a file
    df_rets.to_pickle(os.path.join(cf.DATA_DIR,"sim_port_rets.pickle"))
    
    # Save index weights to a file (pickle and csv)
    weights.to_pickle(os.path.join(cf.DATA_DIR,"cap1_idx_weights.pickle"))
    weights.to_csv(os.path.join(cf.DATA_DIR,"cap1_idx_weights.csv"))
    
    return weights, df_rets, tests

#%% Entry point
if __name__ == "__main__":
    weights, df_rets, tests = main()
    
    