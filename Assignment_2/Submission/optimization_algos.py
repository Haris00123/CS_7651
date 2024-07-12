"""
File: NNnet_trainin.py
Author: Muhammad Haris Masood
Date: February 27, 2024
Description: The script compares several optimization algorithims on various optimization problems
"""

import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
from mlrose_hiive import GeomDecay,ExpDecay,MaxKColorGenerator, KnapsackGenerator
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import time
import multiprocessing as multi
import json

MAX_ATTEMPTS=350
MAX_ITERATIONS=1e4

#MAX_ATTEMPTS=20
#MAX_ITERATIONS=1000

#Function to evaluate an algorithim on a particular problem of size length
def function_eval(fitness_func,fitness_func_name,length,algorithim,algorithim_name,hyperparameter_dict,results_dict,only_binary=True,tune=True,reps=5,graph=True,**kwargs):

    '''
    Function to evaluate an algorithm on a particular problem of size length
    
    Parameters:
    fitness_func (function): The fitness function to be used
    fitness_func_name (str): Name of the fitness function
    length (int): Size of the problem
    algorithim (function): The algorithm to evaluate
    algorithim_name (str): Name of the algorithm
    hyperparameter_dict (dict): Dictionary of hyperparameters to tune
    results_dict (dict): Dictionary to store results
    only_binary (bool): Whether to generate binary problems only
    tune (bool): Whether to tune hyperparameters
    reps (int): Number of repetitions
    graph (bool): Whether to generate graphs
    **kwargs: Additional keyword arguments for fitness function
    
    Returns:
    None
    '''

    best_score=0
    best_iters=0
    best_evals=0
    best_time=0

    MAKKCORRECTION=2

    best_params={}
    os.chdir(os.getcwd())

    #Getting the optimal value for the problem 

    i=0

    if tune:

        for k,v in hyperparameter_dict.items():

            hyperparameter_scores=[]
            hyperparameter_iters=[]
            hyperparameter_evals=[]
            
            for value in v:
                
                t_iter=0
                t_evals=0
                t_score=0

                for ii in range(reps):

                    if only_binary:
                        high=1
                    else:
                        high=length-1

                    if fitness_func_name=='MaxKColor':
                        init_state=np.random.randint(0,high+1,size=MAKKCORRECTION*length)
                    else:              
                        init_state=np.random.randint(0,high+1,size=length)
                    
                    #Init Problem
                    if kwargs:
                        fitness=fitness_func(**kwargs)
                    else:
                        if fitness_func_name!='MaxKColor':
                            fitness=fitness_func()
                        else:
                            pass

                    #Creating the problem
                    if fitness_func_name=='MaxKColor':
                        
                        problem = MaxKColorGenerator().generate(seed=ii,number_of_nodes=length, max_connections_per_node=2*length, max_colors=3)
                        init_state=None

                    elif fitness_func_name=='KnapSack':

                        problem = KnapsackGenerator().generate(seed=ii,number_of_items_types=length)
                        init_state=None
                    
                    else:

                        np.random.seed(ii)

                        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val= high+1)

                    #Evaluation for parameter
                    
                    best_state, best_fitness, curve = algorithim(problem, **{k : value},
                                                max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERATIONS,
                                                init_state = init_state, random_state = ii,curve=True)

                    t_score+=best_fitness

                    # else:   
                    t_iter+=curve.shape[0]
                    t_evals+=curve[-1][1]

                #Averaging score per run
                hyperparameter_scores.append(t_score/reps)
                hyperparameter_iters.append(np.round(t_iter/reps))
                hyperparameter_evals.append(np.round(t_evals/reps))

            i+=1

            hyperparameter_scores=np.array(hyperparameter_scores)
            hyperparameter_iters=np.array(hyperparameter_iters)
            hyperparameter_evals=np.array(hyperparameter_evals)

            #Select Best 
            #if score equal, use the lower function evaluation value

            val=np.argwhere(hyperparameter_scores == np.amax(hyperparameter_scores)).flatten()

            if len(val)>1:
                evals=MAX_ITERATIONS*MAX_ATTEMPTS

                for shortlisted_val in val:
                    if hyperparameter_evals[shortlisted_val]<evals:
                        chosen=shortlisted_val
                        evals=hyperparameter_evals[shortlisted_val]
            else:
                chosen=val[0]
                
            best_params[k]=v[chosen]

            if graph:

                #Creating plot for the score and hyperparameter
                plt.figure();
                plt.title('Score for param : {} for prob: {}, and algo: {} \n Problem Size {}'.format(k,fitness_func_name,algorithim_name,length),fontsize=8)

                if min(hyperparameter_scores)==max(hyperparameter_scores):
                    
                    if min(hyperparameter_scores)==0:
                        low=max(hyperparameter_scores)-0.5
                        high=max(hyperparameter_scores)
                    else:
                        low=max(hyperparameter_scores)*0.75
                        high=max(hyperparameter_scores)
                else:
                    low=min(hyperparameter_scores)
                    high=max(hyperparameter_scores)

                try:
                    plt.scatter(v,hyperparameter_scores)
                    plt.vlines(best_params[k],low,high,label='Best Hyperparameter Value',color='red')
                except TypeError:
                    plt.scatter(list(range(len(v))),hyperparameter_scores)
                    plt.xticks(ticks=list(range(len(v))),labels=[i.__repr__() for i in v],fontsize=8)
                    plt.vlines(list(range(len(v)))[np.argmax(hyperparameter_scores)],low,high,label='Best Hyperparameter Score',color='red')
                plt.ylabel('Function score')
                plt.xlabel('Hyperameter "{}" Value'.format(k))
                plt.legend()
                
                folder_path='p1\{}\{}'.format(fitness_func_name,algorithim_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                folder_path=folder_path+'\{}_{}_size'.format(k,length)
                plt.tight_layout()
                plt.savefig(folder_path);
                plt.close()

    else:

        best_params=hyperparameter_dict.copy()   
    
    #Running function again to get best values

    curves=[]

    for i in range(reps):

        if only_binary:
            high=1
        else:
            high=length-1

        if fitness_func_name=='MaxKColor':
            init_state=np.random.randint(0,high+1,size=MAKKCORRECTION*length)
        else:              
            init_state=np.random.randint(0,high+1,size=length)
        
        #Init Problem
        if kwargs:
            fitness=fitness_func(**kwargs)
        else:
            if fitness_func_name!='MaxKColor':
                fitness=fitness_func()
            else:
                pass
        
        #Creating the problem
        if fitness_func_name=='MaxKColor':
            
            problem = MaxKColorGenerator().generate(seed=i,number_of_nodes=length, max_connections_per_node=2*length, max_colors=3)
            init_state=None

        elif fitness_func_name=='KnapSack':

            problem = KnapsackGenerator().generate(seed=i,number_of_items_types=length)
            init_state=None
            
        else:
            np.random.seed(i*i)
            problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val= high+1)

        start=time.time()

        #Evaluation for parameter
        best_state, best_fitness, curve = algorithim(problem, **best_params,
                                    max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERATIONS,
                                    init_state = init_state, random_state = i,curve=True)

        end=time.time()

        best_score+=best_fitness/reps

        best_iters+=curve.shape[0]/reps

        best_evals+=curve[-1][1]/reps

        best_time+=(end-start)/reps

        curves.append(curve[:,0])

    #Getting average of runs
    final_curves=[]
    max_size=0
    for curve in curves:
        if curve.shape[0]>max_size:
            max_size=curve.shape[0]
    
    for curve in curves:
        final_curves.append(np.pad(curve,pad_width=(0,max_size-curve.shape[0]),mode='edge'))

    final_curve=np.mean(np.array(final_curves),axis=0)

    results = {'best_score':best_score,'best_iters':np.ceil(best_iters),'best_evals':np.ceil(best_evals),'best_time':best_time,'best_state':best_state,'final_curve':final_curve,'best_params':best_params}

    results_dict['overall_iterations'][fitness_func_name][algorithim_name][length]=results['final_curve']
    results_dict['overall_score'][fitness_func_name][algorithim_name][length]=results['best_score']
    results_dict['overall_time'][fitness_func_name][algorithim_name][length]=results['best_time']
    results_dict['overall_evals'][fitness_func_name][algorithim_name][length]=results['best_evals']

def create_results_dict(size_range=range(5,50,10),multi_process=True):

    
    #creating result dictionary

    if multi_process:
    
        manager=multi.Manager()
        overall_iterations=manager.dict()
        overall_score=manager.dict()
        overall_time=manager.dict()
        overall_evals=manager.dict()
        size_range=size_range

        for problem in OPT_PROBLEMS.keys():

            algorithim_iterations=manager.dict()
            algorithim_score=manager.dict()
            algorithim_time=manager.dict()
            algorithim_evals=manager.dict()

            for algorithim in ALGORITHIM.keys():
                
                internal_iteration=manager.dict()
                internal_scores=manager.dict()
                internal_time=manager.dict()
                internal_evals=manager.dict()
                    
                for size in size_range:

                    internal_iteration[size]=0
                    internal_scores[size]=0
                    internal_time[size]=0
                    internal_evals[size]=0

                algorithim_iterations[algorithim]=internal_iteration
                algorithim_score[algorithim]=internal_scores
                algorithim_time[algorithim]=internal_time
                algorithim_evals[algorithim]=internal_evals

            overall_iterations[problem]=algorithim_iterations
            overall_score[problem]=algorithim_score
            overall_time[problem]=algorithim_time
            overall_evals[problem]=algorithim_evals

        results={'overall_iterations':overall_iterations,'overall_score':overall_score,'overall_time':overall_time,'overall_evals':overall_evals}

    else:
    
        #creating result dictionary
        overall_iterations={}
        overall_score={}
        overall_time={}
        overall_evals={}
        size_range=size_range

        for problem in OPT_PROBLEMS.keys():

            algorithim_iterations={}
            algorithim_score={}
            algorithim_time={}
            algorithim_evals={}

            for algorithim in ALGORITHIM.keys():
                
                internal_iteration={}
                internal_scores={}
                internal_time={}
                internal_evals={}
                    
                for size in size_range:

                    internal_iteration[size]=0
                    internal_scores[size]=0
                    internal_time[size]=0
                    internal_evals[size]=0

                algorithim_iterations[algorithim]=internal_iteration
                algorithim_score[algorithim]=internal_scores
                algorithim_time[algorithim]=internal_time
                algorithim_evals[algorithim]=internal_evals

            overall_iterations[problem]=algorithim_iterations
            overall_score[problem]=algorithim_score
            overall_time[problem]=algorithim_time
            overall_evals[problem]=algorithim_evals

        results={'overall_iterations':overall_iterations,'overall_score':overall_score,'overall_time':overall_time,'overall_evals':overall_evals}

    return results
            
if __name__ == "__main__":

    OPT_PROBLEMS={
        'MaxKColor':mlrose.MaxKColor,
        'KnapSack':mlrose.Knapsack,
        'FourPeaks':mlrose.FourPeaks,
        'FlipFlop':mlrose.FlipFlop,
        'Queens':mlrose.Queens,
        'OneMax':mlrose.OneMax,
        'SixPeaks':mlrose.SixPeaks,
        }

    ALGORITHIM={
        'RHC':mlrose.random_hill_climb,
        'SA':mlrose.simulated_annealing,   
        'GA':mlrose.genetic_alg,
        'MIMIC':mlrose.mimic
    }

    HYPERPARAMETER_PARENTS_DICT= {
        'RHC':{
            'restarts':np.arange(0,50,5)
        },
        
        'SA':{
            'schedule': [
            GeomDecay(10),
            GeomDecay(50),
            GeomDecay(100),
            ExpDecay(10),
            ExpDecay(50),
            ExpDecay(100)
            ]   
        },

        'GA':{
            'pop_size':[250,500],
            'pop_breed_percent':[0.1,0.25,0.5],
            'mutation_prob':[0.1,0.25,0.35]
        },

        'MIMIC':{
            'pop_size':[250,500],
            'keep_pct':[0.1,0.25,0.35]
        }
    }

    #Multiprocessing and processing
    tune=True

    #CHANGE PROBLEM SIZE HERE
    size_range=[3,4,5]

    results_dict=create_results_dict(size_range,multi_process=True)

    processes=[]

    for problem in OPT_PROBLEMS.keys():

        for algorithim in ALGORITHIM.keys():

            for size in size_range:
                
                if problem=='KnapSack':
                    values=np.random.randint(1,size,size=size)
                    weights=np.random.randint(1,size,size=size)
                
                if problem=='KnapSack' or problem=='Queens' or problem=='MaxKColor':
                    only_binary=False
                else:
                    only_binary=True

                if problem=='KnapSack':
                    kwargs={'weights':weights,'values':values}
                    p=multi.Process(target=function_eval,args=[
                        OPT_PROBLEMS[problem],problem,size,ALGORITHIM[algorithim],algorithim,HYPERPARAMETER_PARENTS_DICT[algorithim],results_dict,only_binary,tune],kwargs=kwargs)
                    p.start()
                    processes.append(p)
                else:
                    p=multi.Process(target=function_eval,args=[
                        OPT_PROBLEMS[problem],problem,size,ALGORITHIM[algorithim],algorithim,HYPERPARAMETER_PARENTS_DICT[algorithim],results_dict,only_binary,tune])
                    p.start()
                    processes.append(p)
            
    for process in processes:
        process.join()

    #Plotting

    sizes_label={0:'SMALL',1:'MEDIUM',2:'LARGE'}


    folder_path='p1'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    #Saving dictionary in p1
    file_path='p1/results_dict.json'
    with open(file_path, 'w') as fp:
        json.dump(pd.Series(dict(results_dict)).to_json(orient='values'), fp)

    #Splitting results into dictionaries
    overall_iterations=results_dict['overall_iterations']
    overall_score=results_dict['overall_score']
    overall_time=results_dict['overall_time']
    overall_evals=results_dict['overall_evals']

    for problem in OPT_PROBLEMS.keys():

        folder_path='p1\Final\{}'.format(problem)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #Iterations & Fitness Score (With Problem Size) per algorithim per problem
        plt.figure();
        fig,axes=plt.subplots(1,len(size_range))
        fig.set_size_inches(8*len(size_range),10)
        plt.suptitle('Iterations & Average Score For Optimization Problem: "{}"'.format(problem),fontsize=18)

        for i,size in enumerate(size_range):
            
            axes[i].set_title('Problem Size {} "{}" Problem'.format(size,sizes_label[i]))

            for algorithim in ALGORITHIM.keys():

                axes[i].plot(np.arange(overall_iterations[problem][algorithim][size].shape[0]),overall_iterations[problem][algorithim][size],label=algorithim)

            axes[i].set_ylabel('Score');
            axes[i].set_xlabel('Iteration');
            axes[i].legend();

        file_path=folder_path+'\iter_score.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        #Score & Problem Size per algorithim per problem
        plt.figure();
        fig,axes=plt.subplots(1,len(size_range))
        fig.set_size_inches(8*len(size_range),10)
        
        plt.suptitle('Score for Optimization Problem: "{}"'.format(problem),fontsize=18)

        for i,size in enumerate(size_range):
            
            axes[i].set_title('Problem Size {} "{}" Problem'.format(size,sizes_label[i]))

            for ii,algorithim in enumerate(ALGORITHIM.keys()):

                axes[i].bar(algorithim,overall_score[problem][algorithim][size],label=algorithim)
                axes[i].bar_label(axes[i].containers[ii], label_type='edge')

            axes[i].set_ylabel('Score');
            axes[i].set_xlabel('Algorithim');
            axes[i].legend();

        file_path=folder_path+'\score_size.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()


        #Function Evals & Problem Size per algorithim per problem

        plt.figure();
        fig,axes=plt.subplots(1,len(size_range))
        fig.set_size_inches(8*len(size_range),10)
        
        plt.suptitle('Number of Function Evals for Optimization Problem: "{}"'.format(problem),fontsize=18)

        for i,size in enumerate(size_range):
            
            axes[i].set_title('Problem Size {} "{}" Problem'.format(size,sizes_label[i]))

            for ii,algorithim in enumerate(ALGORITHIM.keys()):

                axes[i].bar(algorithim,overall_evals[problem][algorithim][size],label=algorithim)
                axes[i].bar_label(axes[i].containers[ii], label_type='edge')

            axes[i].set_ylabel('Function Evaluations');
            axes[i].set_xlabel('Algorithim');
            axes[i].legend();

        file_path=folder_path+'\\feval_size.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        #Time & Problem Size per algorithim per problem

        plt.figure();
        fig,axes=plt.subplots(1,len(size_range))
        fig.set_size_inches(8*len(size_range),10)
        
        plt.suptitle('Time to converge for Optimization Problem: "{}"'.format(problem),fontsize=18)

        for i,size in enumerate(size_range):
            
            axes[i].set_title('Problem Size {} "{}" Problem'.format(size,sizes_label[i]))

            for ii,algorithim in enumerate(ALGORITHIM.keys()):

                axes[i].bar(algorithim,overall_time[problem][algorithim][size],label=algorithim)
                axes[i].bar_label(axes[i].containers[ii], label_type='edge')

            axes[i].set_ylabel('Time (s)');
            axes[i].set_xlabel('Algorithim');
            axes[i].legend();

        file_path=folder_path+'\\t_size.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()