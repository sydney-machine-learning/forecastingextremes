import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import Flatten
from keras.layers import Conv1D
#from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
#from keras.layers.convolutional import MaxPooling1D
import numpy as np
import time

#from keras.layers import Bidirectional
#from keras.layers import RepeatVector
#from keras.layers import TimeDistributed

from sklearn.metrics import mean_squared_error



def MODEL_LSTM(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):    
    '''
    Trains and evaluates an LSTM using data in x_train and y_train
    
    Args:
        x_train:
        x_test:
        y_train:
        y_test:
        train_params: dictionary containing parameters for training the model
        evaler: Evaluator object for model evaluation
    '''
    
    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']
    
    if forecast_dir and os.path.exists(forecast_dir.joinpath(f"LSTM_{desc}.keras")):
        model_path = forecast_dir.joinpath(f"LSTM_{desc}.keras")
        model = load_model(model_path)
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        evaler.evaluateMetrics(1,desc,y_predicttrain,y_train,y_predicttest,y_test,None)
        best_case_cw = evaler.getMetricScore(1,desc,'CaseWeight')
        best_predict_test = y_predicttest
        best_predict_train = y_predicttrain
        best_model = model
        log_print(f"Loaded forecasting model from {model_path}")
    else:
        model = Sequential()
        model.add(LSTM(hidden, activation='relu', input_shape=(n_steps_in,n_fvars)))
        model.add(Dense(n_steps_out))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        best_case_cw=1000   #Assigning a large number 
        start_time=time.time()
        log_print(f"Starting LSTM forecaster training on {desc}")
        for run in range(1,num_exp+1):
            log_print(f"\tExperiment {run} in progress")
            # fit model
            start_fit = time.time()
            model.fit(x_train, y_train, epochs=epochs,batch_size=64, verbose=0, shuffle=False)
            fit_time = time.time() - start_fit
            y_predicttrain = model.predict(x_train)
            y_predicttest = model.predict(x_test)
            #evaluate results
            evaler.evaluateMetrics(run,desc,y_predicttrain,y_train,y_predicttest,y_test, fit_time)
            curr_test_case = evaler.getMetricScore(run,desc,'CaseWeight')
            if np.mean(curr_test_case) < np.mean(best_case_cw):
                best_case_cw = curr_test_case
                best_predict_test = y_predicttest
                best_predict_train = y_predicttrain
                best_model = model
        
        log_print(f"Total time for {num_exp} {desc} experiments {time.time()-start_time}")
    
    return best_predict_test, best_predict_train, best_case_cw, best_model

def MODEL_CNN(x_train, x_test, y_train, y_test, desc, train_params, evaler, log_print):
    num_exp = train_params['num_exp']
    hidden = train_params['hidden']
    epochs = train_params['epochs']
    forecast_dir = train_params['forecast_dir']
    n_steps_out = train_params['n_steps_out']
    n_steps_in = train_params['n_steps_in']
    n_fvars = train_params['n_fvars']

    #TODO: load pre-trained forecasting model
    #if forecast_dir and os.path.exists(forecast_dir.joinpath(f"LSTM_{desc}.keras")):

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(n_steps_in,n_fvars)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    best_case_cw=1000   #Assigning a large number 
    start_time=time.time()
    log_print(f"Starting CNN forecaster training on {desc}")
    for run in range(1,num_exp+1):
        log_print(f"\tExperiment {run} in progress")
        # fit model
        start_fit = time.time()
        model.fit(x_train, y_train, epochs=epochs,batch_size=64, verbose=0, shuffle=False)
        fit_time = time.time() - start_fit
        y_predicttrain = model.predict(x_train)
        y_predicttest = model.predict(x_test)
        #evaluate results
        evaler.evaluateMetrics(run,desc,y_predicttrain,y_train,y_predicttest,y_test, fit_time)
        curr_test_case = evaler.getMetricScore(run,desc,'CaseWeight')
        if np.mean(curr_test_case) < np.mean(best_case_cw):
            best_case_cw = curr_test_case
            best_predict_test = y_predicttest
            best_predict_train = y_predicttrain
            best_model = model
    log_print(f"Total time for {num_exp} {desc} experiments {time.time()-start_time}")
    
    return best_predict_test, best_predict_train, best_case_cw, best_model

    
#OLD:
def COMBINED_LSTM(x_train,x_res,x_test,y_train,y_res,y_test,Num_Exp,n_steps_in,n_steps_out,Epochs,Hidden):
    n_features = 1
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features))
    print(x_train.shape)
    
    x_res = x_res.reshape((x_res.shape[0], x_res.shape[1], n_features))
    print(x_res.shape)
    
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    print(x_test.shape)
    
    train_acc=np.zeros(Num_Exp)
    test_acc=np.zeros(Num_Exp)
    Step_RMSE=np.zeros([Num_Exp,n_steps_out])
    
    model = Sequential()
    model.add(LSTM(Hidden, activation='relu', input_shape=(n_steps_in,n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    Best_RMSE=1000   #Assigning a large number 
    
    start_time=time.time()
    for run in range(Num_Exp):
        print("Experiment",run+1,"in progress")
        # fit model
        model.fit(x_train, y_train, epochs=Epochs,batch_size=64, verbose=0, shuffle=False)
        y_predicttest1 = model.predict(x_test)
        
        model.fit(x_res, y_res, epochs=Epochs,batch_size=64, verbose=0, shuffle=False)
        y_predicttest2 = model.predict(x_test)
        
        y_combined = y_predicttest2
        for i in range(0,np.shape(y_predicttest1)[0]):
            if(y_predicttest2[i][0] < 0.4):
                y_combined[i][0] = y_predicttest1[i][0]
        test_acc[run] = mean_squared_error( y_combined, y_test) 
        if test_acc[run]<Best_RMSE:
            Best_RMSE=test_acc[run]
            Best_Predict_Test=y_combined
            
    print("Total time for",Num_Exp,"experiments",time.time()-start_time)
    return test_acc,Step_RMSE,Best_Predict_Test
