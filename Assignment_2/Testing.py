import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
from mlrose_hiive import GeomDecay,ExpDecay
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import time
import multiprocessing as multi

MAX_ATTEMPTS=1000
MAX_ITERATIONS=100000

#Function to evaluate an algorithim on a particular problem of size length
def function_eval(fitness_func,fitness_func_name,length,algorithim,algorithim_name,hyperparameter_dict,results_dict,only_binary=True,reps=3,graph=True,**kwargs):

    best_score=0
    best_iters=0
    best_evals=0
    best_time=0

    best_params={}
    os.chdir(os.getcwd())

    #Getting the optimal value for the problem 

    i=0
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
                
                init_state=np.random.randint(0,high+1,size=length)
                
                #Init Problem
                if kwargs:
                    fitness=fitness_func(**kwargs)
                else:
                    fitness=fitness_func()

                #Creating the problem
                problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val= high+1)

                #Evaluation for parameter
                best_state, best_fitness, curve = algorithim(problem, **{k : value},
                                            max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERATIONS,
                                            init_state = init_state, random_state = ii,curve=True)

                t_score+=best_fitness

                # if (curve.shape[0]==curve[-1][1])==min(MAX_ATTEMPTS,MAX_ITERATIONS):
                    
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
            
            folder_path='images\{}\{}'.format(fitness_func_name,algorithim_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            folder_path=folder_path+'\{}_{}_size'.format(k,length)
            plt.tight_layout()
            plt.savefig(folder_path);
            plt.close()
    
    #Running function again to get best values

    for i in range(reps):

        if only_binary:
            high=1
        else:
            high=length-1

        init_state=np.random.randint(0,high+1,size=length)

        #Init Problem
        if kwargs:
            fitness=fitness_func(**kwargs)
        else:
            fitness=fitness_func()

        #Creating the problem
        problem = mlrose.DiscreteOpt(length = length, fitness_fn = fitness, maximize = True, max_val= high+1)

        start=time.time()

        #Evaluation for parameter
        best_state, best_fitness, curve = algorithim(problem, **best_params,
                                    max_attempts = MAX_ATTEMPTS, max_iters = MAX_ITERATIONS,
                                    init_state = init_state, random_state = ii,curve=True)

        end=time.time()

        best_score+=best_fitness/reps

        best_iters+=curve.shape[0]/reps

        best_evals+=curve[-1][1]/reps

        best_time+=(end-start)/reps

    results = {'best_score':best_score,'best_iters':np.ceil(best_iters),'best_evals':np.ceil(best_evals),'best_time':best_time,'best_state':best_state,'final_curve':curve,'best_params':best_params}
    
    results_dict['overall_iterations'][fitness_func_name][algorithim_name][length]=results['final_curve']
    results_dict['overall_score'][fitness_func_name][algorithim_name][length]=results['best_score']
    results_dict['overall_time'][fitness_func_name][algorithim_name][length]=results['best_time']
    results_dict['overall_evals'][fitness_func_name][algorithim_name][length]=results['best_evals']

def create_results_dict(size_range=range(5,50,10),multi_process=True):

    manager=multi.Manager()
    #creating result dictionary
    overall_iterations=manager.dict()
    overall_score=manager.dict()
    overall_time=manager.dict()
    overall_evals=manager.dict()
    size_range=size_range

    if multi_process:
                
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
        'Queens':mlrose.Queens,
        'KnapSack':mlrose.Knapsack
        }

    ALGORITHIM={
        'RHC':mlrose.random_hill_climb,
        'SA':mlrose.simulated_annealing,   
    }

    HYPERPARAMETER_PARENTS_DICT= {
        'RHC':{
            'restarts':np.arange(0,50,5)
        },
        
        'SA':{
            'schedule': [GeomDecay(100),
            ExpDecay(100)]   
        }
    }

    #Multiprocessing and processing

    size_range=range(5,50,25)

    results_dict=create_results_dict(size_range,multi_process=True)

    processes=[]

    for problem in OPT_PROBLEMS.keys():

        for algorithim in ALGORITHIM.keys():

            for size in size_range:
                
                if problem=='KnapSack':
                    values=np.random.randint(1,20,size=size)
                    weights=np.random.randint(1,20,size=size)
                
                if problem=='KnapSack' or problem=='Queens':
                    only_binary=False
                else:
                    only_binary=True

                if problem=='KnapSack':
                    kwargs={'weights':weights,'values':values}
                    p=multi.Process(target=function_eval,args=[
                        OPT_PROBLEMS[problem],problem,size,ALGORITHIM[algorithim],algorithim,HYPERPARAMETER_PARENTS_DICT[algorithim],results_dict,only_binary],kwargs=kwargs)
                    p.start()
                    processes.append(p)
                else:
                    p=multi.Process(target=function_eval,args=[
                        OPT_PROBLEMS[problem],problem,size,ALGORITHIM[algorithim],algorithim,HYPERPARAMETER_PARENTS_DICT[algorithim],results_dict,only_binary])
                    p.start()
                    processes.append(p)
            
    for process in processes:
        process.join()

    #Plotting
    #Splitting results into dictionaries
    overall_iterations=results_dict['overall_iterations']
    overall_score=results_dict['overall_score']
    overall_time=results_dict['overall_time']
    overall_evals=results_dict['overall_evals']

    for problem in OPT_PROBLEMS.keys():

        folder_path='images\Final\{}'.format(problem)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #Iterations & Fitness Score (With Problem Size) per algorithim per problem
        plt.figure();
        fig,axes=plt.subplots(1,len(ALGORITHIM.keys()))
        fig.set_size_inches(35,10)
        plt.suptitle('Iteartions & Score For Optimization Problem: "{}"'.format(problem),fontsize=18)

        for i,algorithim in enumerate(ALGORITHIM.keys()):
            
            axes[i].set_title('Algorithim : "{}"'.format(algorithim))

            for size in size_range:

                axes[i].plot(np.arange(overall_iterations[problem][algorithim][size].shape[0]),overall_iterations[problem][algorithim][size][:,0],label=str(size))

            axes[i].set_ylabel('Score');
            axes[i].set_xlabel('Iteration');
            axes[i].legend();

        file_path=folder_path+'\iter_score.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        #Score & Problem Size per algorithim per problem
        plt.figure();
        fig,axes=plt.subplots(1,len(ALGORITHIM.keys()))
        fig.set_size_inches(35,10)
        plt.suptitle('Score & Problem Size : "{}"'.format(problem),fontsize=18)

        for i,algorithim in enumerate(ALGORITHIM.keys()):
            
            axes[i].set_title('Algorithim : "{}"'.format(algorithim))
            axes[i].plot(list(size_range),list(overall_score[problem][algorithim].values()),label='Score & Problem Size')

            axes[i].set_ylabel('Score');
            axes[i].set_xlabel('Problem Size');
            axes[i].legend();

        file_path=folder_path+'\score_size.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        #Time & Problem Size per algorithim per problem
        plt.figure();
        fig,axes=plt.subplots(1,len(ALGORITHIM.keys()))
        fig.set_size_inches(35,10)
        plt.suptitle('Time & Problem Size : "{}"'.format(problem),fontsize=18)


        for i,algorithim in enumerate(ALGORITHIM.keys()):
            
            axes[i].set_title('Algorithim : "{}"'.format(algorithim))
            axes[i].plot(list(size_range),list(overall_time[problem][algorithim].values()),label='Time & Problem Size')

            axes[i].set_ylabel('Time');
            axes[i].set_xlabel('Problem Size');
            axes[i].legend();

        file_path=folder_path+'\\t_size.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

        #Function Evaluations per alogorithim per problem 

        plt.figure();
        fig,axes=plt.subplots(1,len(ALGORITHIM.keys()))
        fig.set_size_inches(35,10)
        plt.suptitle('Num Function Evals & Problem Size : "{}"'.format(problem),fontsize=18)

        for i,algorithim in enumerate(ALGORITHIM.keys()):
            
            axes[i].set_title('Algorithim : "{}"'.format(algorithim))
            axes[i].plot(list(size_range),list(overall_evals[problem][algorithim].values()),label='Function Evals & Problem Size')

            axes[i].set_ylabel('Num Function Evals');
            axes[i].set_xlabel('Problem Size');
            axes[i].legend();

        file_path=folder_path+'\\feval_size.png'
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        

