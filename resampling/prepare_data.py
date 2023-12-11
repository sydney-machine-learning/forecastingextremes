import pandas as pd
import numpy as np

def get_taken_df(df, n_steps_in, n_steps_out, fvars, n_fvars, tvar):
    df = pd.DataFrame(df)
    cols, names = list(), list()
    #input sequence (t-n_steps_in, ..., t-1)
    for i in range(n_steps_in, 0, -1):
        cols.append(df.shift(i))
        names+= [f"{fvars[j]}(t-{i})" for j in range(n_fvars)]
    #forecast sequence (t, t+1, ..., t+n_steps_out)
    for i in range(0, n_steps_out):
        cols.append(df[0].shift(-i))
        if i == 0:
            names += [f"{tvar}(t)"]
        else:
            names += [f"{tvar}(t+{i})"]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg

def get_X_cols(cols):
    return [item for item in cols if "t-" in item]

def get_Y_cols(cols):
    return [item for item in cols if "t-" not in item]
    #def contains_any(string, substrings):
    #    return any(substring in string for substring in substrings)
    #return [item for item in Ys if contains_any(item, targets)]


def split_taken(taken, n_steps_in, n_steps_out, fvars, n_fvars, tvar, split_ratio):
    
    training_size = int(taken.shape[0]*split_ratio)
    test_size = taken.shape[0] - training_size

    train_frame = taken[:training_size]
    test_frame = taken[training_size:]

    X_cols = get_X_cols(taken.columns)
    Y_cols = get_Y_cols(taken.columns)

    X_train_df = train_frame[X_cols]
    Y_train_df = train_frame[Y_cols]
    X_test_df = test_frame[X_cols]
    Y_test_df = test_frame[Y_cols]

    X_train = X_train_df.values
    X_train = X_train.reshape((X_train.shape[0], n_steps_in, n_fvars))

    X_test = X_test_df.values
    X_test = X_test.reshape((X_test.shape[0], n_steps_in, n_fvars))

    Y_train = Y_train_df.values
    Y_train = Y_train.reshape((Y_train.shape[0], n_steps_out))
    
    Y_test = Y_test_df.values
    Y_test = Y_test.reshape((Y_test.shape[0], n_steps_out))
    
    k_X = X_train.reshape(X_train.shape[0], n_steps_in*n_fvars)
    k_X = np.concatenate((k_X,Y_train),axis=1)
    
    return k_X, X_train, Y_train, X_test, Y_test

def split_data(data, n_steps_in, n_steps_out, fvars, n_fvars, tvar, split_ratio):
    taken = get_taken_df(data, n_steps_in, n_steps_out, fvars, n_fvars, tvar)
    
    return split_taken(taken, n_steps_in, n_steps_out, fvars, n_fvars, tvar, split_ratio)

def split_cyclone_data(scaled_data, cyclone_labels, N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR, SPLIT_RATIO):
    '''Applies taken embedding to each individual cyclone separately and concats all results
    '''
    id = cyclone_labels[0]
    start_index = 0
    end_index = 0
    taken = pd.DataFrame()
    skipped_cyclones = []
    for i in range(1, scaled_data.shape[0]):
        if cyclone_labels[i] == id:
            end_index+=1
        else:
            #cyclone is only usable if there are >= N_STEPS_IN + N_STEPS_OUT many samples 
            if (N_STEPS_IN + N_STEPS_OUT) <= (end_index - start_index + 1):
                curr_taken = get_taken_df(scaled_data[start_index:end_index+1], N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR)
                taken = pd.concat([taken, curr_taken], axis=0)
            else:
                skipped_cyclones.append(cyclone_labels[i-1])
            '''
            if (end_index - start_index + 1) == (N_STEPS_IN + N_STEPS_OUT) + 1:
                print(f"Exactly size 11 cyclone sequence at {cyclone_labels[i-1]}, {i}")
                print(f"Scaled data: {scaled_data[start_index:end_index+1]}")
                print(f"Embedded: {curr_taken}")
            '''   
            #    print(f"FAIL cyc: {i} has start: {start_index}, end: {end_index}")
            id = cyclone_labels[i]
            start_index=i
            end_index=i
        if i == scaled_data.shape[0]-1:
            #cyclone is only usable if there are >= N_STEPS_IN + N_STEPS_OUT many samples 
            curr_taken = get_taken_df(scaled_data[start_index:end_index+1], N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR)
            taken = pd.concat([taken, curr_taken], axis=0)
    #print(f"Skipped cyclones: {skipped_cyclones}")
            
    return split_taken(taken, N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR, SPLIT_RATIO)

def split_time_series(data, DATA_NAME, N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR, SPLIT_RATIO):
    if DATA_NAME == "Cyclone":
        return split_cyclone_data(data[:,0], data[:,1], N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR, SPLIT_RATIO)
    else:
        return split_data(data, N_STEPS_IN, N_STEPS_OUT, FVARS, N_FVARS, TVAR, SPLIT_RATIO)