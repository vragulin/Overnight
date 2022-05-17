# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:34:54 2022
Load Fed Funds rates from Fred and generate monthly collateral returns
Save into a pickle file
@author: vragu
"""

import pandas as pd
import numpy as np
import fredpy as fp
import matplotlib.pyplot as plt
import pickle
plt.style.use('classic')


FF_HIST_PICKLE = '../data/FF_hist.pickle'
FRED_API_KEY   = '../../Bond Predictability/FRED_API_Key.txt'

#%% Load Fed Funds history from FRED
def load_ff_from_FRED(reload=True, save=True):
    """ Load full Fed Funds history from FERD
        Params: reload - if True from FRED, else from pickle
                save - save new result to pickle if reload=True
        Output: pandas series of daily FF rates
        
    """
    
    if reload:
        # Initialize FRED API Key
        fp.api_key = fp.load_api_key(FRED_API_KEY)
    
        # Download data from FRED
        ff_series = fp.series('DFF').data
    
        # Save into a pickle file
        if save:
            with open(FF_HIST_PICKLE, "wb") as f:
                pickle.dump(ff_series, f)
                
    else:
        # Load file from pickle
        with open(FF_HIST_PICKLE, "rb") as f:
            ff_series = pickle.load(f)
            
    return ff_series

#%% Generate a history of FF Deposit cum returns (inverse discount factors)
def gen_ff_returns(ff_series):
    """ Generate a series of cumulative returns on a Fed Funds deposit account
    Parameters: ff_series : TYPE - pandas.series of Fed Funds rates
    Returns: pandas series with cum return (future value) of a FF depo account
    """

    # FF use actual/360 convention
    return (1 + ff_series / 36000).cumprod()

#%% Generate periodic discount factor and cont. compounding returns
def gen_period_returns(daily_cumrets, period='M', start=None, end=None):
    """ Generate a series of periodic cumulative and period returns on a Fed Funds deposit account
    Parameters: 
        daily_cumrets - pandas.series of Fed Funds cumulative returns
        period - string code of period
        start  - start date (if 'None', match daily_cumrets)
        end    - end date (if 'None', match daily_cumrets)
        
    Returns: pandas series with cum return (future value) of a FF depo account
    """
    
    # Resample daily deposit factor to the desired frequency
    # For now we ignore incomplete periods at the start and end - this can be fixed later
    cumrets = daily_cumrets.resample(period).last()
    df = pd.DataFrame(cumrets)
    df.columns = ['Cum_Rets']

    # Select only data within the desired period
    if start:
        if end:
            df = df[start:end]
        else:
            df = df[start:]
    else:
        if end:
            df = df[:end]

    df.Cum_Rets = df.Cum_Rets / df.Cum_Rets.iloc[0]    
    df['Rets'] = np.log(df.Cum_Rets).diff()
    
    return df
     
# =============================================================================
# Start of Main
# =============================================================================
if __name__ == "__main__":
    
    # Load FF
    ff_series = load_ff_from_FRED(reload=False)
    
    print(ff_series.head())
    print(ff_series.tail())
    ff_series.plot()
    plt.show()
    
    # Calculate value of a FF deposit account
    rets = gen_ff_returns(ff_series)
    print(rets.head())
    print(rets.tail())
    
    rets.plot()
    plt.show()
    
    # Test function of generating periodic returns
    df = gen_period_returns(rets, period='M', start='1993-01-01', end=None)
    print(df.head())
    print(df.tail())
    
    
    