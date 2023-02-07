import pandas as pd
import numpy as np
from copy import deepcopy


def prepare_data( df, features, n_steps=4, with_intersection=False, flatten=False ):
    '''
        [1...n] rows as features, row n+1 as label
        The next feature starts at row n+2 (no intersection) or row 2 ([2...n+1])
    '''
    
    _df = df.copy()
    n = n_steps                                                          # easier to read code    
        
    assert isinstance(_df, pd.core.frame.DataFrame) and len(df) > n, 'Incorrect input dataframe provided'
    assert isinstance(features, list) and len(features) > 0 and isinstance(features[0], (str, int)), 'Incorrect features'
    assert isinstance(n, int) and n < len(_df), 'Incorrect number of steps'
           
    # GET THE LAST DATA POINT FOR FINAL VERIFICATION
    golden_set  = _df.tail( n+1 )
    goldenx     = golden_set[ features ].head( n ).values.tolist()
    if flatten:
        goldenx = [ i for sublist in goldenx for i in sublist ]
    goldeny     = golden_set[ features ].tail(1).values.tolist()[0]
      
    if with_intersection:
        # DROP LAST DATA POINT (last row)
        print('Dropping 1 last row for golden data point')
        _df = _df.head( len(_df) - 1 ).reset_index(drop=True)

    else:    
        # DROP LAST DATA POINT (last n+1 rows)
        print('Dropping {} last rows for golden data point'.format( n+1 ))
        _df = _df.head( len(_df) - n - 1 ).reset_index(drop=True)
        
        # DROP ROWS FROM BEGINNING IF _df.shape[0] % (n+1) != 0
        offset = len(_df) % (n + 1)
        print('Shape of data: {}. Dropping {} first rows because n+1 = {}'.format( _df.shape, offset, n+1 ))
        _df = _df.tail( len(_df) - offset ).reset_index(drop=True)
        
    # ITERATE OVER REMAINING ROWS TO GET X AND Y
    X, y = [], []
    i = 0
        
    if with_intersection:
        constraint = len(_df) - n
    else:
        constraint = len(_df)
                
    while i < constraint:
        this_x = _df[ features ].iloc[ i:i+n ].values.tolist()
        if flatten:
            this_x = [ i for sublist in this_x for i in sublist ]
        this_y = _df[ features ].iloc[ i+n  ].values.tolist()
        X.append( this_x )
        y.append( this_y )
                
        if with_intersection:
            i += 1
        else:
            i = i + n + 1
    
    print('Data prepared:')
    print('\tSize of X = ({} by {})'.format( len(X), len(X[0] )))
    print('\tSize of y = ({} by {})'.format( len(y), len(y[0])))
    _df = None
    
    return ( np.array(X),
             np.array(y),
             np.array(goldenx),
             np.array(goldeny)
           )



def prepare_data_intersection( df, features, n=4 ):
    '''
        [1...n] rows as features, row n+1 as label
        The next feature starts at 
    '''
    
    assert isinstance(df, pd.core.frame.DataFrame) and len(df) > n, 'Incorrect input dataframe provided'
    assert isinstance(features, list) and len(features) > 0 and isinstance(features[0], (str, int)), 'Incorrect features'
    assert isinstance(n, int) and n < len(df), 'Incorrect number of steps'
    
    _df = df.copy()
        
    # GET THE LAST DATA POINT FOR FINAL VERIFICATION
    golden_set = _df.tail( n+1 )
    goldenx    = golden_set[ features ].head( n ).values.tolist()
    goldenx    = [ i for sublist in goldenx for i in sublist ]
    goldeny    = golden_set[ features ].tail(1).values.tolist()[0]
        
    # DROP LAST DATA POINT (last row)
    print('Dropping 1 last row for golden data point')
    _df = _df.head( len(_df) - 1 ).reset_index(drop=True)
            
    # ITERATE OVER DF TO GET X AND Y
    X, y = [], []
    i = 0
    while i < ( len(_df) - n ):
        this_x = _df[ features ].iloc[ i:i+n ].values.tolist()
        this_x = [ j for sublist in this_x for j in sublist ]
        this_y = _df[ features ].iloc[ i+n   ].values.tolist()
        X.append( this_x )
        y.append( this_y )
        i += 1
    
    print('Data prepared:')
    print('\tSize of X = ({} by {})'.format( len(X), len(X[0] )))
    print('\tSize of y = ({} by {})'.format( len(y), len(y[0])))
    _df = None
    
    return X, y, goldenx, goldeny


# n_samples = n_samples to generate; defaults to ALL if None
def train_test_shuffle_split( X, y, test_size=0.2 ):
    '''
        Shuffle X, y, then do train / test split
    '''
        
    # NUMBER OF DATA POINT IN TEST SET
    num_points = int( len(y)*test_size )
    print('\nTrain test split: {}%/{}%'.format( 100-(test_size*100), test_size*100 ))
    X, y = shuffle( X, y, random_state=random_state, n_samples=None )
    return X[ : -num_points ], X[ -num_points : ], y[ : -num_points ], y[ -num_points : ]



def train_test_seq_split( X, y, test_size=0.2 ):
    '''
        Do train / test split without shuffling (time series)
    '''
        
    # NUMBER OF DATA POINT IN TEST SET
    num_points = int( len(y)*test_size )
    print('\nTrain test split: {}%/{}%'.format( 100-(test_size*100), test_size*100 ))                    
    return X[ : -num_points ], X[ -num_points : ], y[ : -num_points ], y[ -num_points : ]


def print_folds_stats( X, X_sh, tscv, kf ):
    
    X_   = deepcopy( X )
    X_sh_ = deepcopy( X_sh )
    
    if tscv:
        print('\nTimeSeriesSplit indices:')
        for train_idx, test_idx in tscv.split( X_ ):
            print('(Train, test): ({:>3}, {})'.format( len(train_idx),  len(test_idx) ))
            
    if kf:
        print('\nKFold indices:')
        for train_idx, test_idx in kf.split( X_sh_ ):
            print('(Train, test): ({:>3}, {})'.format( len(train_idx),  len(test_idx) ))