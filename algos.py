import math
import copy

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

import torch
from kan import KAN

from tools import KAN_es


def create_XY_data(source_data, output_parameter, geophysical_method, samples_number = "all"):
    
    _source_data = copy.deepcopy(source_data)
    _output_parameter = copy.deepcopy(output_parameter)
    _geophysical_method = copy.deepcopy(geophysical_method)
    _samples_number = copy.deepcopy(samples_number)
    
    
    if not(_output_parameter in ['H1_8', 'H2_8', 'H3_8']):
        print("Error: output parameter is not in the list")
        return 
    
    if not(_geophysical_method in ['G', 'M', 'T', 'GM', 'GT', 'MT','GMT']):
        print("Error: physical_method is not in the list!")
        return
    
    if _samples_number == "all":
        _samples_number = len(_source_data)
    
    #TODO: Create checking sourse data
    
    if _geophysical_method == 'G':
        _X_data = _source_data.iloc[:_samples_number, :31]
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]
    elif _geophysical_method == 'M':
        _X_data = _source_data.iloc[:_samples_number, 31:62]
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]        
    elif _geophysical_method == 'T':
        _X_data = _source_data.iloc[:_samples_number, 62:124]
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]
    elif _geophysical_method == 'GM':
        _X_data = _source_data.iloc[:_samples_number, :62]
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]
    elif _geophysical_method == 'GT':
        _X_data = pd.concat([_source_data.iloc[:_samples_number, :31], 
                             _source_data.iloc[:_samples_number, 62:124]], 
                             axis=1, sort=False, ignore_index=False)
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]              
    elif _geophysical_method == 'MT':
        _X_data = _source_data.iloc[:_samples_number, 31:124]
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]       
    elif _geophysical_method == 'GMT':
        _X_data = _source_data.iloc[:_samples_number, :124]
        _Y_data = _source_data[[_output_parameter]].iloc[:_samples_number,:]
                        
    return _X_data, _Y_data   


def train_NN(trn_data, vld_data, 
             geophysical_method, output_parameter, samples_number, randomseed, 
             model_name_template):
    
    _trn_data = copy.deepcopy(trn_data) 
    _vld_data = copy.deepcopy(vld_data)
    _geophysical_method = copy.deepcopy(geophysical_method)
    _output_parameter = copy.deepcopy(output_parameter)    
    _samples_number = copy.deepcopy(samples_number)  
    _randomseed = copy.deepcopy(randomseed)
    _model_name_template = copy.deepcopy(model_name_template)
    
    
    _model_name = (_model_name_template + "_" + 
                   _geophysical_method + "_" + 
                   str(_samples_number) + "_" +
                   _output_parameter + "_rs" +
                   str(_randomseed))
                               
    ### Create input-output data
    
    _dataset_sizes = {"all":"all", 3500:1000, 1750:500, 1000:300, 700:200, 350:100, 175:50}
    _X_trn, _Y_trn = create_XY_data(_trn_data, _output_parameter, _geophysical_method, _samples_number)
    _X_vld, _Y_vld = create_XY_data(_vld_data, _output_parameter, _geophysical_method, _dataset_sizes[_samples_number])
    
    ### Create model
    
    tf.keras.utils.set_random_seed(_randomseed)
    
    _model = tf.keras.models.Sequential() 
    _model.add(tf.keras.layers.Dense(32, activation=tf.nn.sigmoid))
    _model.add(tf.keras.layers.Dense(1,  activation=None))    

    _optimizer = keras.optimizers.SGD(learning_rate = 0.1,#0.001,
                                      momentum = 0.5,
                                      nesterov=True)
        
    _model.compile(_optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    _early_stopping = EarlyStopping(monitor='val_loss',
                                    min_delta=0.001 * 0.05,#0.0001, 
                                    patience=500,#500,
                                    restore_best_weights=True)
      
    ### Train model
    print("Start train model:", _model_name)

    _history = _model.fit(x=_X_trn, y=_Y_trn,
                          batch_size=math.ceil(len(_X_trn) * 0.05),
                          epochs=50000, #100000,
                          verbose=0,
                          callbacks=[_early_stopping],
                          validation_data=(_X_vld, _Y_vld))

    _n_epochs = len(_history.history['loss'])
    
    print('Number of epochs for training NN model:' + str(_n_epochs))
    
    ### Save model
    
    #_model.save(_dir_path+"/"+ _model_name+".keras")
    
    _history_df = pd.DataFrame(_history.history)
    #plot_history_NN(_history_df, _model_name, _dir_path)
    
    #_history_df.to_csv(_dir_path+"/"+ _model_name + '_history.csv')
    
    return _model, _history_df, _n_epochs


def app_stat_NN(model, tst_data, 
                geophysical_method, output_parameter, samples_number, randomseed, 
                file_name_template, file_name_suffix):
        
    _tst_data = copy.deepcopy(tst_data) 
    _geophysical_method = copy.deepcopy(geophysical_method)
    _output_parameter = copy.deepcopy(output_parameter)
    _samples_number = copy.deepcopy(samples_number) 
    _randomseed = copy.deepcopy(randomseed)    
    _file_name_template = copy.deepcopy(file_name_template)
    _file_name_suffix = copy.deepcopy(file_name_suffix)    
    #_dir_path = copy.deepcopy(dir_path)
    
    _file_name =  (_file_name_template + "_" + 
                   _geophysical_method + "_" + 
                   str(_samples_number) + "_" +
                   _output_parameter + "_rs" +
                   str(_randomseed) + "_" + 
                   file_name_suffix)
      
    ### Create input-output data
    
    _X_tst, _Y_tst = create_XY_data(_tst_data, _output_parameter, _geophysical_method, "all")
  
    ### Apply and calculate statistics

    _Y_pred = model[0].predict(_X_tst).reshape(-1, 1)
    
    _mae = round(mean_absolute_error(_Y_tst, _Y_pred), 5)
    _rmse = round(mean_squared_error(_Y_tst, _Y_pred, squared = False), 5)
    _r2 = round(r2_score(_Y_tst, _Y_pred), 5)



    #_statistics_DF = pd.DataFrame({'epochs': model[2]-500, 'MAE': _mae, 'RMSE':_rmse, 'R2': _r2}, 
    #                              index=[_file_name])
    
    #print(_statistics_DF)
    
    #_statistics_DF.to_csv(_dir_path+"/"+ _file_name + '.csv')
    
    #return _statistics_DF
    return [_rmse, _mae, _r2]


def alg_keras_mlp(trn_data, vld_data, tst_data,
             geophysical_method, output_parameter, randomseed=None
             ):
    
    keras_nn = train_NN(trn_data, vld_data,
                         geophysical_method,
                         output_parameter, 
                         "all", randomseed, 
                         "?")
            
    stat_DF = app_stat_NN(keras_nn, tst_data,
                      geophysical_method,
                      output_parameter, 
                      "all", randomseed, 
                      "stat_udus", "?")
    
    return stat_DF


def vector_pred_NN(trn_data, vld_data, tst_data,
                geophysical_method, l_output_parameter, randomseed=None, 
                model_name_template='?'):
    '''Conducting prediction for each output from l_output_parameter. 
    One NN for each output_parameter from l_output_parameter.
    Return vector [mae, mse, r2]
    '''
    _geophysical_method = copy.deepcopy(geophysical_method)
    _tst_data = copy.deepcopy(tst_data)

    _X_tst, _Y_tst = create_XY_data(_tst_data, l_output_parameter[0], _geophysical_method, "all")

    _vector_Y_pred = np.array([[] for i in range(_Y_tst.shape[0])])
    _vector_Y_tst = np.array([[] for i in range(_Y_tst.shape[0])])

    for output_parameter in l_output_parameter:
        model = train_NN(trn_data, vld_data, 
                geophysical_method, output_parameter, "all", randomseed, 
                model_name_template=model_name_template)
        
        
        _output_parameter = copy.deepcopy(output_parameter)
        _X_tst, _Y_tst = create_XY_data(_tst_data, _output_parameter, _geophysical_method, "all")
        _Y_pred = model[0].predict(_X_tst).reshape(-1, 1)

        _vector_Y_pred = np.concatenate((_vector_Y_pred, _Y_pred), axis=1)
        _vector_Y_tst = np.concatenate((_vector_Y_tst, _Y_tst), axis=1)

    _mae = round(mean_absolute_error(_vector_Y_tst, _vector_Y_pred), 5)
    _rmse = round(mean_squared_error(_vector_Y_tst, _vector_Y_pred, squared = False), 5)
    _r2 = round(r2_score(_vector_Y_tst, _vector_Y_pred), 5)

    return [_mae, _rmse, _r2]


# ----- KAN -----


def get_KAN_dataset(trn_data, vld_data, tst_data,
                    geophysical_method, output_parameter, samples_number
                    ):
    
    _trn_data = copy.deepcopy(trn_data)
    _vld_data = copy.deepcopy(vld_data)
    _tst_data = copy.deepcopy(tst_data)
    _geophysical_method = copy.deepcopy(geophysical_method)
    _output_parameter = copy.deepcopy(output_parameter)    
    _samples_number = copy.deepcopy(samples_number)

    ### Create input-output data

    _dataset_sizes = {"all":"all", 3500:1000, 1750:500, 1000:300, 700:200, 350:100, 175:50}
    _X_trn, _Y_trn = create_XY_data(_trn_data, _output_parameter, _geophysical_method, _samples_number)
    _X_vld, _Y_vld = create_XY_data(_vld_data, _output_parameter, _geophysical_method, _dataset_sizes[_samples_number])
    _X_tst, _Y_tst = create_XY_data(_tst_data, _output_parameter, _geophysical_method, 'all')

    _tc_X_trn, _tc_Y_trn = torch.from_numpy(_X_trn.to_numpy()), torch.from_numpy(_Y_trn.to_numpy())
    _tc_X_vld, _tc_Y_vld = torch.from_numpy(_X_vld.to_numpy()), torch.from_numpy(_Y_vld.to_numpy())
    _tc_X_tst, _tc_Y_tst = torch.from_numpy(_X_tst.to_numpy()), torch.from_numpy(_Y_tst.to_numpy())

    dataset_3 = {'train_input': _tc_X_trn,
             'train_label': _tc_Y_trn,
             'val_input': _tc_X_vld,
             'val_label': _tc_Y_vld,
             'test_input': _tc_X_tst,
             'test_label': _tc_Y_tst}
    
    return dataset_3


def train_KAN(dataset_3,
                RS=1,
                K=3, # order of piecewise polynomial in B-splines
                hidden_neurons=1
                ):
    
    ### Create model
    INPUT_SHAPE = dataset_3['train_input'].shape[1]

    model = KAN_es(width=[INPUT_SHAPE, hidden_neurons, 1], grid=3, k=K, seed=RS)

    result = model.train_es(dataset_3,
                          tol=0.001, #0.0001
                          n_iter_no_change=25,
                          opt="LBFGS", steps=1000, 
                          lamb=0,
                          lamb_l1=1,
                          lamb_entropy=2)
    
    return model, result


def alg_KAN_model(trn_data, vld_data, tst_data,
                  geophysical_method, output_parameter, randomseed=None,
                  K=3,hidden_neurons=1
                  ):
    dataset_3 = get_KAN_dataset(trn_data, vld_data, tst_data,
                geophysical_method,
                output_parameter, 
                "all")
    
    model = train_KAN(dataset_3, randomseed, K=K, hidden_neurons=hidden_neurons)

    _Y_tst = dataset_3['test_label'].detach().numpy()
    _Y_pred = model[0].forward(dataset_3['test_input']).detach().numpy()

    _mae = round(mean_absolute_error(_Y_tst, _Y_pred), 5)
    _rmse = round(mean_squared_error(_Y_tst, _Y_pred, squared = False), 5)
    _r2 = round(r2_score(_Y_tst, _Y_pred), 5)

    return [_mae, _rmse, _r2]


def vector_pred_KAN(trn_data, vld_data, tst_data,
                    geophysical_method, l_output_parameter, randomseed=None, 
                    K=3,hidden_neurons=1):
    '''Conducting prediction for each output from l_output_parameter. 
    One KAN for each output_parameter from l_output_parameter.
    Return vector [mae, mse, r2]
    '''
    dataset_3 = get_KAN_dataset(trn_data, vld_data, tst_data,
                        geophysical_method, l_output_parameter[0], 'all')

    _vector_Y_pred = np.array([[] for i in range(dataset_3['test_label'].shape[0])])
    _vector_Y_tst = np.array([[] for i in range(dataset_3['test_label'].shape[0])])


    for output_parameter in l_output_parameter: 
        dataset_3 = get_KAN_dataset(trn_data, vld_data, tst_data,
                        geophysical_method, output_parameter, 'all')
        
        model = train_KAN(dataset_3,
                          RS=randomseed,
                          K=K,
                          hidden_neurons=hidden_neurons)
        
        _Y_tst = dataset_3['test_label'].detach().numpy()
        _Y_pred = model[0].forward(dataset_3['test_input']).detach().numpy()
        
        _vector_Y_pred = np.concatenate((_vector_Y_pred, _Y_pred), axis=1)
        _vector_Y_tst = np.concatenate((_vector_Y_tst, _Y_tst), axis=1)
        
    _mae = round(mean_absolute_error(_vector_Y_tst, _vector_Y_pred), 5)
    _rmse = round(mean_squared_error(_vector_Y_tst, _vector_Y_pred, squared = False), 5)
    _r2 = round(r2_score(_vector_Y_tst, _vector_Y_pred), 5)

    return [_mae, _rmse, _r2]


# ----- Multi Exp------

def multi_exp(l_algos_names,
              l_algos,
              mult_data,
              l_geophysical_method,
              l_output_parameter,
              l_kwargs,
              l_metrics_names,
              num_iter):
    ''' Function, that process algos(X, Y) and returns df of their metrics. 
    '''
    res_list = []

    for alg, (trn, val, test), kwargs, alg_name, geophysical_method, output_parameter in zip(l_algos, 
                                                                                             mult_data, 
                                                                                             l_kwargs, 
                                                                                             l_algos_names, 
                                                                                             l_geophysical_method, 
                                                                                             l_output_parameter):
        print(f'--- Processing {alg_name}')

        for i in range(1, num_iter+1):
            print(f'iter: {i}')
            #print(kwargs)
            l_metrics = alg(trn, val, test, geophysical_method, output_parameter, randomseed=i, **kwargs)
            res_list.append([alg_name, i]+l_metrics)
        print('-------')

    return pd.DataFrame(res_list, columns=['alg_name', 'iter']+l_metrics_names)