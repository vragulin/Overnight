# -*- coding: utf-8 -*-
"""
Created on Tue May 24 18:32:49 2022
Config file for API Keys etc.
@author: vragu
"""

# storing API data
#AV_API_KEY = "SLDCIR0NMNO87Y48"  - non-Premium (Free) key
AV_API_KEY = "GI075O91R1R45N6F"

# eodhistoricaldata defaults
EOD_API_KEY = "629855d0595fc2.43499588"
EOD_HISTORICAL_DATA_API_KEY_DEFAULT = EOD_API_KEY
EOD_HISTORICAL_DATA_API_URL = "https://eodhistoricaldata.com/api"

# directory structure
DATA_DIR     = "../data"
SPLIT_DIR    = "../split_dfs"
STOCK_DIR    = "../stock_dfs"
DIV_DIR      = "../div_dfs"
SHRS_OUT_DIR = "../shrs_dfs"
MERGED_DIR   = "../merged_dfs"
HSTAT_DIR    = "../hist_stats"
SIM_STK_DIR = "../sim_stk_dfs"

# Global variables - file locations
TICKER_FILE    = "elm_good_tkrs.pickle"
ALL_STOCKS_DFS = "elm_stocks_px_mcap.pickle"