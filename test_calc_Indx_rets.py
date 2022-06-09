# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 07:56:33 2022
Test ca;c_index_return function
@author: vragu
"""

import numpy as np
import pandas as pd

from build_index_sim_stocks import calc_index_returns

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
                 [0,        1      ],
                 [np.nan,   1      ] ] 

    df_weights = pd.DataFrame(weights3, columns = df_close.columns)    
    df = calc_index_returns(df_close, df_weights)
    print(df)