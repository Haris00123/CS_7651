from NNnet_Support import nn_support
from sklearn.model_selection import train_test_split,KFold,LearningCurveDisplay
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import time
from mlrose_hiive import NeuralNetwork,GeomDecay,ExpDecay
import mlrose_hiive
import numpy as np
import multiprocessing as multi
import os


def deal_param(data,param,param_val,algorithm,default_dict,metric=accuracy_score,metric_name='Accuracy',graph=True,cv=3):

    '''Function to create parameter curves
    
    Parameters:
    data (list): List of np.arrays containing the features and labels  
    param (str): Parameter to vary
    param_vals (list): Parameter values
    ax (matplotlib.axes): Axis for graph
    algorithm (sklearn.algorithm/NNnet): Algorithm to vary parameter in
    metric (sklearn.metric): Metric to score algorotihim on
    metric_name (str): Metric Name
    graph (bool): Bool for validation graph
    cv (int): The number of cross validations folds
    
    Returns:
    None'''

    train_acc=[]
    test_acc=[]
    os.chdir(os.getcwd())

    #Crossvalidating each parameter value
    kf=KFold(n_splits=cv,shuffle=True)

    internal_train_accuracy=0
    internal_test_accuracy=0

    internal_dict=default_dict.copy()

    if param in internal_dict.keys():
        internal_dict[param]=param_val
    else:
        internal_dict={param:param_val,**internal_dict}
    
    param_dict={'algorithm':algorithm,**internal_dict}

    c=0
    for train,test in kf.split(X=data[0]):

        np.random.seed(c)
        c+=1
        clf=NeuralNetwork(**param_dict)

        _=clf.fit(data[0][train],data[1][train])
        internal_train_accuracy+=metric(y_pred=clf.predict(data[0][train]),y_true=data[1][train])
        internal_test_accuracy+=metric(y_pred=clf.predict(data[0][test]),y_true=data[1][test])

    train_acc.append(internal_train_accuracy/cv)
    test_acc.append(internal_test_accuracy/cv)

    print('For param value : {} for param : {} the average accuracy for train was {} was for test was {}'.format(param,param_val,train_acc[0],test_acc[0]))
    
    return None

if __name__ == "__main__":

    #Using support class to get dataset
    support=nn_support()

    #Loading Data
    heart=support.load_heart()

    #Splitting into test & train
    train,test=support.split_data(X=heart[0],Y=heart[1],valid=False)

    #Standardizing
    train_standardized,test_standardized=support.standardize_data(train,test)

    default_dict={
        'max_attempts':50,
        'max_iters':1e5,
        #'max_iters':100,
        'learning_rate':0.001,
        'clip_max':1000,
        'nodes':50,
        'layers':1,
        'restarts':35,
        'activation':'relu',
        'curve':True,
        'early_stopping':True
    }

    param='learning_rate'

    val_dist = {
        'random_hill_climb':{
            'learning_rate':[1e-3,1e-2,1e-1,0.5],
        },
    }
    

    for param_val in val_dist:
        p=multi.Process(target=deal_param,args=[
                        train_standardized,param,param_val,algorithm,default_dict,])
        p.start()
        processes.append(p)
            
    for process in processes:
        process.join()


    


