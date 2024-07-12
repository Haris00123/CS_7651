"""
File: NNnet_training.py
Author: Muhammad Haris Masood
Date: February 27, 2024
Description: The script comples the neural net portion of assignment_2
"""

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
ALGO_DICTS = {
        'random_hill_climb':{
            'restarts':np.arange(0,50,5)
        },
        
        'simulated_annealing':{
            'schedule': [
            GeomDecay(5),
            GeomDecay(10),
            GeomDecay(50),
            ExpDecay(5),
            ExpDecay(10),
            ExpDecay(50)]   
        },

        'genetic_alg':{
            'pop_size':[250,500,1000],
            'pop_breed_percent':[0.1,0.25,0.5],
            'mutation_prob':[0.1,0.25,0.35]
        },

}

def create_param_curve(data,param,param_vals,ax,algorithm,default_dict,metric=accuracy_score,metric_name='Accuracy',graph=True,cv=5):

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

    for i in param_vals:
        #Crossvalidating each parameter value
        kf=KFold(n_splits=cv,shuffle=True)

        internal_train_accuracy=0
        internal_test_accuracy=0

        internal_dict=default_dict.copy()

        if param in internal_dict.keys():
            internal_dict[param]=i
        else:
            internal_dict={param:i,**internal_dict}
        
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

    best_val=param_vals[np.argmax(test_acc)]

    if type(param_vals[0])==mlrose_hiive.algorithms.decay.geom_decay.GeomDecay:
        ax.scatter(list(range(len(param_vals))),train_acc,label='Training {}'.format(metric_name))
        ax.scatter(list(range(len(param_vals))),test_acc,label='Validation {}'.format(metric_name))
        ax.set_xticks(ticks=list(range(len(param_vals))),labels=[i.__repr__() for i in param_vals],fontsize=8)
        ax.axvline(list(range(len(param_vals)))[np.argmax(test_acc)],label='Best {} Value'.format(param),color='red',linestyle = '--')
    else:
        ax.scatter(param_vals,train_acc,label='Training {}'.format(metric_name))
        ax.scatter(param_vals,test_acc,label='Validation {}'.format(metric_name))
        ax.axvline(best_val,label='Best {} Value'.format(param),color='red',linestyle = '--')
    ax.legend()
    ax.set_ylabel('{}'.format(metric_name));
    ax.set_xlabel(param);
    ax.set_title('{} vs {}'.format(param,metric_name));
    
    return best_val

def deal_algorithm(data,test,param_dicts,dataset,algorithm,default_dict,metric=accuracy_score,metric_name='Accuracy',cv=5):
    
    '''Function to deal with algorithm
    
    Parameters:
    data (list): List of np.arrays containing the features and labels
    param_dict (dict): Parameter dictionary to vary
    dataset (str): Dataset name
    algorithm_name (str): Algorithm name
    algorithm (sklearn.algorithm/NNnet): Algorithm to vary parameter in
    metric (sklearn.metric): Metric to score algorotihim on
    metric_name (str): Metric Name
    cv (int): The number of cross validations folds 

    Returns:
    None'''
    
    if algorithm=='gradient_descent':
        adder=2
    else:
        adder=3
        #adder=2

    num_classes=len(param_dicts.keys())+adder
    best_vals={}

    #Getting Fig Size
    fig,axes=plt.subplots(1,num_classes)
    fig.set_size_inches(8*num_classes,10)
    i=-1
    for c,ax in enumerate(fig.axes[:-adder]):
        i+=1
        plt.suptitle('Results for Neural Net with: "{}" for "{}" Dataset'.format(algorithm,dataset),fontsize=18)
        param=list(param_dicts.keys())[i]
        param_vals=param_dicts[param]
        best_val=create_param_curve(data,param,param_vals,ax,algorithm,default_dict,metric,metric_name,cv=cv)
        best_vals[param]=best_val

    final_dict=default_dict.copy()
    return_dict=default_dict.copy()
    
    for i in best_vals.keys():    
        final_dict[i]=best_vals[i]
        return_dict[i]=best_vals[i]

    final_dict['algorithm']=algorithm

    #Final Function Evaluation
    
    #Get Time taken
    #Get Train & Test Score
    #Plot average loss curve
    #Metrics for final learner
    train_time=0
    training_accuracy=0
    test_accuracy=0

    c=0
    for i in range(cv):

        np.random.seed(c)

        c+=1

        start_training=time.time()

        #Creating best learner
        learner=NeuralNetwork(**final_dict)

        ##Learning
        learner.fit(data[0],data[1])

        end_training=time.time()

        train_time+=(end_training-start_training)/cv

        #Predictions
        preds_train=learner.predict(data[0])
        preds_test=learner.predict(test[0])
        
        #Training Accuracy
        training_accuracy+=accuracy_score(data[1],preds_train)/cv

        #Test Accuracy
        test_accuracy+=accuracy_score(test[1],preds_test)/cv

        #Averaging loss curve
        if algorithm=='gradient_descent':
            axes[-2].plot(learner.fitness_curve,label='loss',alpha=0.5)
            axes[-2].set_title('Loss & Iterations')
            axes[-2].set_xlabel('Iterations')
            axes[-2].set_ylabel('Loss')
            axes[-2].legend()
        else:
            axes[-2].scatter(range(learner.fitness_curve.shape[0]),learner.fitness_curve[:,0],label='loss for run {}'.format(i),alpha=0.5)
            axes[-2].set_title('Loss & Iterations')
            axes[-2].set_xlabel('Iterations')
            axes[-2].set_ylabel('Loss')
            axes[-2].legend()

            axes[-3].scatter(learner.fitness_curve[:,1],learner.fitness_curve[:,0],label='loss for run {}'.format(i),alpha=0.5)
            axes[-3].set_title('Loss & Function Evaluations')
            axes[-3].set_xlabel('Function Evaluations')
            axes[-3].set_ylabel('Loss')
            axes[-3].legend()

    #Learning Curve
    learner=NeuralNetwork(**final_dict)
    LearningCurveDisplay.from_estimator(learner,data[0],data[1],ax=axes[-1])

    plt.tight_layout()
    #Saving Figure
    folder_path='p3\\Neural_Net\{}'.format(algorithm)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path=folder_path+'\experimentation.png'
    plt.savefig(file_path)
    plt.close()


    #Saving Values
    results_dict={
        'training_accuracy':training_accuracy,
        'test_accuracy':test_accuracy,
        'training_time':train_time
    }

    pd.DataFrame(results_dict.items()).to_csv(folder_path+'\\results.csv')
    
    print(default_dict)

    if algorithm=='gradient_descent':
        return return_dict

if __name__ == "__main__":



    default_dict={
        'max_attempts':50,
        'max_iters':1e4,
        #'max_iters':100,
        'learning_rate':0.0001,
        'clip_max':1000,
        'nodes':100,
        'layers':2,
        'activation':'relu',
        'curve':True,
        'early_stopping':True
    }

    #Using support class to get dataset
    support=nn_support()

    #Loading Data
    heart=support.load_heart()

    #Splitting into test & train
    
    train,test=support.split_data(X=heart[0],Y=heart[1],valid=False)

    #Standardizing
    train_standardized,test_standardized=support.standardize_data(train,test)

    #Parameter dict for gradient descent
    param_dict={
        'nodes':[50,100],
        'layers':[1,2],
        'learning_rate':[1e-3,1e-2],
        'clip_max':[1e3,1e4]
    }

    # param_dict={
    #     'nodes':[5]
    # }

    default_dict=optimized_defaults_dict=deal_algorithm(train_standardized,test_standardized,param_dict,'Heart Dataset','gradient_descent',default_dict,cv=5)
    
    processes=[]

    ALGO_DICTS = {
        'random_hill_climb':{
            'learning_rate':[1e-2,1e-1,0.5],
            'restarts':np.arange(0,50,5)
        },
        
        # 'simulated_annealing':{
        #     'learning_rate':[1e-2,1e-1,0.5],
        #     'schedule': [
        #     GeomDecay(5),
        #     GeomDecay(10),
        #     GeomDecay(50),
        #     GeomDecay(100),]   
        # },

        # 'genetic_alg':{
        #     'learning_rate':[1e-2,1e-1,0.5],
        #     'pop_size':[250,500],
        #     'mutation_prob':[0.1,0.25,0.35]
        # },

}
    # default_dict['max_iters']=1e5

    for algorithm in ALGO_DICTS.keys():
        p=multi.Process(target=deal_algorithm,args=[
                        train_standardized,test_standardized,ALGO_DICTS[algorithm],'Heart Dataset',algorithm,default_dict,])
        p.start()
        processes.append(p)
            
    for process in processes:
        process.join()


    # ALGO_DICTS = {
    #     'random_hill_climb':{
    #         'max_attempts':[2,5],
    #         'restarts':[2,5]
    #     },

    #     'simulated_annealing':{
    #         'schedule':[
    #         GeomDecay(5),
    #         GeomDecay(10)]
    #     }
    # }

    # for algorithm in ALGO_DICTS.keys():
    #     p=multi.Process(target=deal_algorithm,args=[
    #                     train_standardized,test_standardized,ALGO_DICTS[algorithm],'Heart Dataset',algorithm,default_dict])
    #     p.start()
    #     processes.append(p)