# -*- coding: utf-8 -*-
"""
Created on Sun May 15 20:50:29 2022

Simulate a strategy of going long/short portfolio of top/bottom deciles based on OBP

@author: vragu
"""

#import os
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import load_FF_rates as ff

#%% Global variables - file locations
TICKER_FILE         = "../data/sp500tickers.pickle"
ALL_STOCKS_DF       = "../data/all_stocks_px_ret.pickle"
STOCK_DFS_DIR       = "../stock_dfs"
MONTHLY_FIELD_DF    = "../data/monthly_{field}.pickle"

#%% Initialize Simulation Parameters
def init_sim_params():
    """ Initialize Sim Params, return dictionary"""
    params = {}
    params['rebuild'      ] = False #Regenerate stock return files
    params['window'       ] = 24  #window in months for the return calculation
    params['trx_costs'    ] = 0  #one-way trading costs in bp
    params['borrow_fee'   ] = 0  #borrow fee on the shorts, in bp per annum
    params['capital'      ] = 1000000  #initial capital
    params['trade_pctiles'] = [20, 80]  #Sell and buy  thresholds for shorts/longs portfolios

    return params

#%% Load all stock panel data from pickle files
def load_data():
    
    with open(TICKER_FILE, "rb") as f:
        tickers = pickle.load(f)
    
    with open(ALL_STOCKS_DF, "rb") as f:
        df_all_stocks = pickle.load(f)

    return tickers, df_all_stocks


#%% Build a monthly dataframe of overnight returns
def load_monthly_df(field = 'r_ovnt', rebuild=False, save_df=True):
    """ Parameters:
            field   - name of a field that we want to load for every stock
            rebuild - if True, generate monthly o/n return from panel data
                     else, load from a pickle file
            save_df - save the resulting dataframe into a pickle
    """
    
    pickle_fname = MONTHLY_FIELD_DF.format(field=field)
    
    if not rebuild:
        # Load file from pickle
        with open(pickle_fname, "rb") as f:
            dfm = pickle.load(f)
    else:
        
        # Re-generate monthy on returns form panel data
        tickers, df_all_stocks = load_data()
        df1 = df_all_stocks.pivot_table(values='Value', index='Date', columns=['Ticker', 'Field'])
            
        # Extract Intraday Returns for all stocks
        df2 = df1.swaplevel(axis=1)[field]
        
        # Resample to monthly frequency
        dfm = df2.resample('M').sum()

        # Save new df into a pickle
        if save_df:
            with open(pickle_fname, "wb") as f:
                pickle.dump(dfm, f)
            
    return dfm

#%% Generate positions for the simulation based on a sort indicator
def gen_positions(df_sort, sim_params):
    """ Generate stock positions
        Params:
            df_sort - dataframe with sort indicator for each date an stock
            sim_params - dictionary with sort parameter
       Return: 
           positions  - dataframe with positions (+1/0/-1) for all stocks and dates
           thresholds - dataframe with values of thresholds used to form percentiles
    """

    # Unpack parameters
    window = sim_params['window'] #window over which we calc obp
    
    # Generate top and bottom quantiles on the basis of the sorting parameter
    trade_pctiles  = sim_params['trade_pctiles'] #Sell and buy  thresholds for shorts/longs portfolios
    
    # Generate positions
    #OBP thresholds long/short portfolios
    thresholds = pd.DataFrame(np.nan, index = df_sort.index, columns=['buy','sell', 'n_buys', 'n_sells'])

    #Positions in each stock at the close of a given date close
    positions  = pd.DataFrame(np.nan, index = df_sort.index, columns = df_sort.columns)
    
    for t, date in enumerate(df_sort.index):
        if t < window-1:
            continue
        
        row  = df_sort.loc[date,:].values
        non_zero_rets = row[row != 0]
        
        # Calculate thresholds for long/short portfolio
        t_sell, t_buy = np.percentile(non_zero_rets, trade_pctiles)
        
        # Calculate positions for all stocks
        positions.loc[date,:] = np.where(df_sort.loc[date,:]>=t_buy,1,0)
        positions.loc[date,:] = np.where(df_sort.loc[date,:]<=t_sell,-1, positions.loc[date,:])
        
        # Calculate the number of longs and shorts
        n_buys  = (positions.loc[date,:] ==  1).values.sum()
        n_sells = (positions.loc[date,:] == -1).values.sum()
        
        print(f"t={t}, t_buy={t_buy:.3f}, t_sell={t_sell:.3f}, n_buys={n_buys}, n_sells={n_sells}")

        thresholds.loc[date,:] = t_buy, t_sell, n_buys, n_sells
        
    return positions, thresholds
    
#%% Calculate portfolio returns from positions and stock returns
def calc_portfolio_returns(pos, ret, ret_full, riskfree, params, intra=False):
    """ Calculate portfolio returns
        Parameters:
            pos      - dataframe of positions for each period
            ret      - dataframe of partial-day returns (o/n or intraday) for each period
            ret_full - dataframe of full day returns
            riskfree - risk-free rate returns (same index as other dataframes)
            params   - dictionary with parameters
            intra    - boolean, True is the strategy does not hold positions overnight
            
        Return: pandas dataframe 
    """

    # Unpack parameters
    try:
        window    =  params['window']
        borrow_fee = params['borrow_fee']
        trx_costs  = params['trx_costs']
    except KeyError as err:
        print("Missing simulation parameter: ", err)
    
    # Set up a dataframe to hold the results
    port_cols  = ['r_l', 'r_s', 'r_ls', 'r_ix', 'r_ix_f']
    port_r     = pd.DataFrame(np.nan, index = ret.index, columns = port_cols)
    port_r_net = pd.DataFrame(np.nan, index = ret.index, columns = port_cols)
    
    # Convert exponential to simlpe returns, so we can aggregate the cross-section
    ret_s      = np.exp(ret)      - 1
    ret_full_s = np.exp(ret_full) - 1
    
    # Populate dataframe with portfolio returns
    for t, date in enumerate(ret.index):
        if t <= window-1:
            continue
    
        # Extract position and return rows
        pos_row = pos.iloc[t-1,:]
        ret_row = ret_s.iloc[t,:]
        ret_full_row = ret_full_s.iloc[t,:]

        # Identify long and short stocks, and stocks with missing data
        long_mask  = ( pos_row ==  1 )
        short_mask = ( pos_row == -1 )
        zero_mask  = ( ret_row ==  0 ) & (ret_full_row == 0)
        
        # Gross returns - first on each stock, then on long/sort portfolio
        stock_r = pos_row * ret_row
        
        port_r.loc[date,'r_l' ]   = stock_r[long_mask].mean()

        if not intra:  #if we hold positions overnight
            port_r.loc[date,'r_l' ]   = stock_r[long_mask].mean()
            port_r.loc[date,'r_s' ]   = stock_r[short_mask].mean() + riskfree.loc[date,'Rets'] * 2
            port_r.loc[date,'r_ix']   = ret_row[~zero_mask].mean()
        else:  #no positions overnight - pay no interest or borrow fees
            port_r.loc[date,'r_l' ]   = stock_r[long_mask].mean()  + riskfree.loc[date,'Rets']           
            port_r.loc[date,'r_s' ]   = stock_r[short_mask].mean() + riskfree.loc[date,'Rets'] 
            port_r.loc[date,'r_ix']   = ret_row[~zero_mask].mean() + riskfree.loc[date,'Rets'] 
            
        port_r.loc[date,'r_ls']   = port_r.loc[date,'r_l'] + port_r.loc[date,'r_s']
        
        # Index buy-and-hold return - subtract initial transactions costs, otherwise assume no rebal
        # Only consider stocks for which we have returns (i.e. ones that trade)
        port_r.loc[date,'r_ix_f'] = ret_full_row[~zero_mask].mean()

        # Caclulate net returns - assume we trade 2x per day on each position
        # Position changes
        stock_r_net     = stock_r - trx_costs * 2 / 10000

        if not intra:
            port_r_net.loc[date,'r_l' ] = stock_r_net[long_mask].mean()
            port_r_net.loc[date,'r_s' ] = stock_r_net[short_mask].mean() + riskfree.loc[date,'Rets'] * 2 \
                                            - borrow_fee / 12 / 10000

        else: #no positions overnight - pay no interest or borrow fees
            port_r_net.loc[date,'r_l' ] = stock_r_net[long_mask].mean()  + riskfree.loc[date,'Rets'] 
            port_r_net.loc[date,'r_s' ] = stock_r_net[short_mask].mean() + riskfree.loc[date,'Rets'] 
                                        

        port_r_net.loc[date,'r_ix'] = port_r.loc[date,'r_ix'] - trx_costs * 2 / 10000
        port_r_net.loc[date,'r_ls'] = port_r_net.loc[date,'r_l'] + port_r_net.loc[date,'r_s']
    
        # Index buy-and-hold return
        if t == window:
            port_r_net.loc[date,'r_ix_f'] = port_r.loc[date,'r_ix_f'] - trx_costs/10000
        else:
            port_r_net.loc[date,'r_ix_f'] = port_r.loc[date,'r_ix_f']

    return port_r, port_r_net
    
#%% Generate cumulative return plots
def plot_sim_returns(ret, use_log=True, start=None, end=None, title_codes=None):
    
    # Extract returns over the focus period
    ret1     = ret[start:end].dropna()
    
    # Cumulative returns
    ret_cum     = (1+ret1).cumprod()
    rln_cum     =  np.log(ret_cum)

    if use_log:
        title_log_str  = "(Log) "
    else:
        title_log_str  = ""

    # Set up labels for the plot    
    legend_template = ['Long {}','Short {}','L/S {}', 'Index {}', 'Index BuyHold']
    linestyles  = ['-','-','-', ':',':']
    
    
    if sim['trade_pctiles'][0] == 20:
        pctile_str = "Top/Bottom Quintiles"
    elif sim['trade_pctiles'][0] == 10:
        pctile_str = "Top/Bottom Deciles"
    else:
        pctile_str = "{}/{} Quantiles".format(*sim['trade_pctiles'])
        
    title_string = "Equity Curves {} - {} - {}\n{} on Trailing (O/N-Intraday), {}"\
                    .format(title_log_str, title_codes[2], title_codes[0], \
                            pctile_str, title_codes[1])
    
    if use_log:
        ax = rln_cum.plot(title = title_string, style=linestyles, fontsize='small')
    else:
        capital = sim['capital']
        ax = (ret_cum*capital).plot(title = title_string, style=linestyles, fontsize='small', logy=True)

    if title_codes[0] == 'Overnight':
        period_label = 'O/N'
    else:
        period_label = 'Intra'
        
    legend_list = [ x.format(period_label) for x in legend_template ] 
    ax.legend(legend_list, fontsize='small')
    ax.set_ylabel("Capital $", fontsize='small')
    ax.set_xlabel("Date", fontsize='small')
    plt.show()
    
    # Return cumulative log returns
    return rln_cum

#%% Generate time series statistics
def get_time_series_stats(rets, riskfree, annualize = True):
    """ Generate descriptive statistics for a time series
        Params: rets     - pandas series exponential period returns
                riskfree - pandas series of exponential riskfree returns
                
        Output: tuple: Mean Ret, Geom Ret, Vol, Sharpe
    """
    
    # Calculate a series of excess return
    xret   = rets - riskfree
    xret_s = np.exp(xret) - 1
    
    xr_mean   = xret_s.mean()
    xr_geom   = np.exp(xret.mean()) - 1
    xr_vol    = xret.std()
    xr_sharpe = xr_mean / xr_vol
    
    if annualize:
        xr_mean   *= 12
        xr_geom   *= 12
        xr_vol    *= np.sqrt(12)
        xr_sharpe *= np.sqrt(12)
        
    return xr_mean, xr_geom, xr_vol, xr_sharpe
    
#%% Generate summary report
def gen_summary_report():
    """ Don't pass any parameters, get all info from the namespace
        Assume that the global namespace contains frames referenced below with cumulative strategy returns
    """
    
    # Set up a dataframe to hold the results
    df_stats = pd.DataFrame(np.nan, index=range(18), \
                            columns = ['Period', 'Window', 'Series', 'Mean Ret', 'Geom Ret', 'Vol', 'Sharpe'])
    
    # Set up iterables
    frames  = [ r_o_pre  ,  r_i_pre  ,  r_o_post,    r_i_post   ]
    periods = ['Pre-2015', 'Pre-2015', 'Post-2015', 'Post-2015' ]
    windows = ['O/N'     , 'Intra'   , 'O/N'      , 'Intra'     ]
    
    series_names = ['Long', 'Short', 'L/S' , 'Index']
    col_names    = ['r_l' , 'r_s'  , 'r_ls', 'r_ix' ]
    stats_names  = df_stats.columns[3:]
    
    i = 0
    for frame, period, window in zip(frames, periods, windows):
        # Add 'active' strategy returns
        for s_name, s_col in zip(series_names, col_names):
            
            # Pick a return series out of an input simulation dataframe
            s = frame[s_col]
                        
            # Calculate stats for the return series (need to diff since s is cumulative)
            stats = get_time_series_stats(s.diff(), riskfree['Rets'])
            
            # Populat a row  in the summary dataframe
            df_stats.loc[i, ['Period', 'Window', 'Series']] = [period, window, s_name]
            df_stats.loc[i, stats_names] = stats
            
            # Increment row counter
            i += 1            
            
    # Add rows for the index buy and hold returns
    for j in [0,2]:
        
        frame, period = frames[j], periods[j]
        
        # Pick the series with the buy-and-hold index return
        s_name = 'Index'
        s_col  = 'r_ix_f'
        window = 'BH'
        
        # Pick a return series out of an input simulation dataframe
        s = frame[s_col]
    
         # Calculate stats for the return series (need to diff since s is cumulative)
        stats = get_time_series_stats(s.diff(), riskfree['Rets'])
         
        # Populat a row  in the summary dataframe
        df_stats.loc[i, ['Period', 'Window', 'Series']] = [period, window, s_name]
        df_stats.loc[i, stats_names] = stats
         
        # Increment row counter
        i += 1            
    
    # Print results
    print(df_stats)
        
    return df_stats         
    
    
#%% Start of the main program
if __name__ == "__main__":
    
    # Initialize simulation parameters
    sim = init_sim_params()
    
    # Generate dataframes of monthly overnight and intraday returns
    df_o = load_monthly_df(field = 'r_ovnt', rebuild=sim['rebuild'])
    df_i = load_monthly_df(field = 'r_intr', rebuild=sim['rebuild'])
    df_f = df_o + df_i #full day return
 
    # Load Fed Funds returns (the risk-free rate)
    riskfree  = ff.load_FF_period_rets(reload=sim['rebuild'],start=df_o.index[0],end=df_o.index[-1])
    
    # Calculate the sorting parameter, for now just use (o/n - intra) window, later can do opb as in Lachance
    df_obp = (df_o-df_i).rolling(sim['window']).sum()
    
    # Generate positions and thresholds used for long and short portfolio cutoffs
    positions, _  = gen_positions(df_obp, sim) 

    # Calculate portfolio monthly and cumulative returns, with and w/o trans costs
    port_o, port_o_net = calc_portfolio_returns(positions, df_o, df_f, riskfree, sim)
    port_i, port_i_net = calc_portfolio_returns(positions, df_i, df_f, riskfree, sim, intra=True)
   
    
    #%% Plot returns for the overnight holding strategy
    use_log = False
 
    # Before 2015
    r_o_pre  = plot_sim_returns(port_o_net, end='2015-01-01', title_codes = ['Overnight','Net', 'Pre-2015'], use_log = use_log) 
    r_i_pre  = plot_sim_returns(port_i_net, end='2015-01-01', title_codes = ['Intraday', 'Net', 'Pre-2015'], use_log = use_log) 
    
    # After 2015
    r_o_post = plot_sim_returns(port_o_net, start='2015-01-01', title_codes = ['Overnight','Net', 'Post-2015'], use_log = use_log) 
    r_i_post = plot_sim_returns(port_i_net, start='2015-01-01', title_codes = ['Intraday', 'Net', 'Post-2015'], use_log = use_log) 

    #%% Analyze the cumulative return time series     
    df_stats = gen_summary_report()
    
    df_stats.to_clipboard()
    print("Done")
    