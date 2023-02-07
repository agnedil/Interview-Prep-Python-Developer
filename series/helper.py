import pandas as pd
import numpy as np
from copy import deepcopy



def prepare_data( df_, features, n_steps=4, overlap=False, flatten=False, y_pos=None, ):
    '''
        Rows [1...n] as features, row n+1 as label (or 1 num from that row if y_pos != None) - n here means n_steps;
        Next feature = starts with row n+2 if not overlap OR row 2 ([2...n+1]) if overlap;
        features - names of feature columns;
        overlap  - False if rows in X are unique (next features starts w/row n+2 as explained above => smaller size of X)
                   OR TRUE if different variables in X can have repeating rows in the rolling fashion
                   (next feature starts w/row 2 as explained above)
        flatten  - True if array of arrays in X should be flatten into one array
    '''
    
    df_ = df_.copy()
    n = n_steps                                                          # easier to read code    
        
    assert isinstance(df_, pd.core.frame.DataFrame), 'Incorrect input dataframe format'
    assert len(df_) > n,                             'Input dataframe size too small'
    assert isinstance(features, list),               'Features should be a list'
    assert len(features) > 0,                        'There should be at least one feature'
    assert isinstance(features[0], (str, int)),      'There should be strings or integers inside each list in features'
    assert isinstance(n, int),                       'Number of steps must be an integer'
    assert n < len(df_),                             'Number of steps too big'
      
    if not overlap:
        # DROP ROWS FROM BEGINNING IF df_.shape[0] % (n+1) != 0
        offset = len(df_) % (n + 1)
        print(f'Total num rows: {df_.shape[0]}, n+1 = {n+1} => dropping {offset} first rows\n')
        df_ = df_.tail( len(df_) - offset ).reset_index(drop=True)
        
    # ITERATE OVER REMAINING ROWS TO GET X AND Y
    X, y = [], []
    i = 0
        
    if overlap:
        max_size = len(df_) - n
    else:
        max_size = len(df_)
                
    '''
    Explaining max_size values:
    * i points to first row of every variable in X    
    * if overlap, last variable in X should start from row len(df)-(n+1) (n_feats+1 row for label),
      then we increase i by 1      =>  i=len(df)-n  =>  execution stops  (no more X, y pairs left)
    * if not everlap, last variable in X should start from row len(df)-(n+1) too (n_feats+1 row for label),
      then we increase i by n + 1  =>  i=len(df)  =>  execution stops    (no more X, y pairs left)
    '''
    while i < max_size:
        this_x = df_[ features ].iloc[ i:i+n ].values.tolist()
        if flatten:
            this_x = [ i for sublist in this_x for i in sublist ]
        this_y = df_[ features ].iloc[ i+n  ].values.tolist()
        X.append( this_x )
        if y_pos is None:
            y.append( this_y )
        else:
            y.append( this_y[ y_pos ])
                
        if overlap:
            i += 1
        else:
            i = i + n + 1

    df_ = None
    
    return ( np.array(X),
             np.array(y),
           )



# n_samples = n_samples to generate; defaults to ALL if None
def train_test_shuffle_split( X, y, test_size=0.2 ):
    '''
        Shuffle X, y, then do train / test split
    '''        
    # NUMBER OF DATA POINTS IN TEST SET
    num_points = int( len(y)*test_size )
    #print('\nTrain test split: {}/{}'.format( 1-test_size, test_size ))
    X, y = shuffle( X, y, random_state=random_state, n_samples=None )
    return X[ : -num_points ], X[ -num_points : ], y[ : -num_points ], y[ -num_points : ]



def train_test_seq_split( X, y, test_size=0.2 ):
    '''
        Train / test split without shuffling (time series)
    '''        
    # NUMBER OF DATA POINTS IN TEST SET
    num_points = int( len(y)*test_size )
    #print('\nTrain test split: {}/{}'.format( 1-test_size, test_size ))                    
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
            
            

            
########################### REFERENCE ONLY #################################



def prepare_data_old( df, features, n_steps=4, with_intersection=False, flatten=False ):
    '''
        OLDER VERSION - DOESN'T USE Y_POS
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