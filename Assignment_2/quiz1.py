import numpy as np
import sys
import random

def optimize_1():
    def f(x):
        #return (x%6)**2%7-np.sin(x)
        return -(5 * x**4 - 13 * x**3 + 2 * x**2 - x)
    
    def convert_int(bits):
        tot=0
        for i,val in enumerate(bits[::-1]):
            tot=tot+(2**i)*val
        return tot

    def generate_population(pop_size=10000):
        return [np.random.randint(0,2,size=7) for i in range(pop_size)]

    def score_pop(population):
        return [f(convert_int(bits)) for bits in population]

    def selected_parents(population,score,size=100):
        parents=[]
        for _ in range(len(population)):
            tournament=random.sample(list(enumerate(population)),size)
            winner=max(tournament,key=lambda x:score[x[0]])
            parents.append(winner[1])
        return parents

    def generate_offspring(parents):
        offspring=[]
        for i in range(0,len(parents),2):
            parent_1,parent_2=parents[i],parents[i+1]
            crossover=random.randint(1,len(parent_1)-1)

            child_1=np.r_[parent_1[:crossover],parent_2[crossover:]]
            child_2=np.r_[parent_2[:crossover],parent_1[crossover:]]
            offspring.extend([child_1,child_2])
        return offspring
    
    def mutate(offspring,mutation_rate=0.2):
        for i in range(len(offspring)):
            if random.random()<mutation_rate:
                mutation_point=random.randint(0,len(offspring[i])-1)
                offspring[i][mutation_point]=np.random.randint(0,2)
        return offspring

    population=generate_population()

    for i in range(10):
        #Scoring
        score=score_pop(population)

        #Selecting parents
        parents=selected_parents(population, score)

        #Generating offspring
        offspring=generate_offspring(parents)

        #Mutation
        offspring=mutate(offspring)

        #switching
        population=offspring
    print(population)
    return convert_int(max(zip(population,score),key=lambda x:x[1])[0])

def optimize_2(x=10,alpha=0.0000001):
    def f(x):
        return -x**4+1000*x**3-20*x**2+4*x-6
    
    def f_p(x):
        return (-4*x**3+3000*x**2-40*x+4)
    
    for i in range(10000):
        x=x+alpha*f_p(x)
    return x

if __name__ == "__main__":
    print(optimize_1())
    print(optimize_2(int(sys.argv[1])))
    
