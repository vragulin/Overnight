# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 07:56:33 2022
Test ca;c_index_return function
@author: vragu
"""

import numpy as np
import pandas as pd

from build_index_sim_stocks import calc_index_returns
from build_index_sim_stocks import calc_mcap_weights


#%% Entry point
if __name__ == "__main__":
    
    close_data = [ [1,        1 ] ,
                   [2,        1  ],
                   [np.nan,   2  ],
                   [3,      np.nan] ] 
    
    df_close = pd.DataFrame(close_data, columns=['AAPL','MSFT'])

    # Test with equal weights
    weights1 = np.ones(df_close.shape) / df_close.shape[1]
    
    df_weights = pd.DataFrame(weights1, columns = df_close.columns)    
    df = calc_index_returns(df_close, df_weights)
    print(df)
    
    # Another test with manually-defined weights
    weights2 = [ [0.5,      0.5,   ],
                 [1,        np.nan ],
                 [1,        0      ],
                 [np.nan,   np.nan ] ] 

    df_weights = pd.DataFrame(weights2, columns = df_close.columns)    
    df = calc_index_returns(df_close, df_weights)
    print(df)

   # Another test with manually-defined weights
    weights3 = [ [0.5,      0.5    ],
                 [1,        np.nan ],
                 [0.5,      0.5   ],
                 [np.nan,   1      ] ] 

    df_weights = pd.DataFrame(weights3, columns = df_close.columns)    
    df = calc_index_returns(df_close, df_weights)
    print(df)
    
    # Test with index
    idx_members = [ [ True, True ],
                    [ True, True ],
                    [ True, True ],
                    [ True, True ] ]

    mcaps    = weights3

    df_idx_memb = pd.DataFrame(idx_members, columns = df_close.columns)   
    df_mcaps    = pd.DataFrame(mcaps, columns = df_close.columns)   
    
    df_weights = calc_mcap_weights(df_mcaps, idx_members = df_idx_memb)
    df = calc_index_returns(df_close, df_weights)
    print(df)
    
    # Another test
    idx_members = [ [ True, False ],
                    [ True, False ],
                    [ True, False ],
                    [ True, True  ] ]

    df_idx_memb = pd.DataFrame(idx_members, columns = df_close.columns)   
    df_weights = calc_mcap_weights(df_mcaps, idx_members = df_idx_memb)
    df = calc_index_returns(df_close, df_weights)
    print(df)
    
    