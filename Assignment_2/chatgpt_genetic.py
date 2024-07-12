import random

# Define the objective function
def objective_function(x):
    return 2 * x

# Generate initial population of integers within a specified range
def generate_population(size, min_value, max_value):
    return [random.randint(min_value, max_value) for _ in range(size)]

# Evaluate the fitness of each individual in the population
def evaluate_population(population):
    return [objective_function(x) for x in population]

# Select parents for mating using tournament selection
def tournament_selection(population, fitness, tournament_size):
    selected_parents = []
    for _ in range(len(population)):
        tournament = random.sample(list(enumerate(population)), tournament_size)
        winner = max(tournament, key=lambda x: fitness[x[0]])
        selected_parents.append(winner[1])
    return selected_parents

# Perform crossover to create offspring
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    return offspring

# Perform mutation on offspring
def mutate(offspring, mutation_rate, min_value, max_value):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            mutation_point = random.randint(0, len(offspring[i]) - 1)
            offspring[i][mutation_point] = random.randint(min_value, max_value)
    return offspring

# Genetic algorithm to maximize the function
def genetic_algorithm(population_size, num_generations, tournament_size, mutation_rate, min_value, max_value):
    # Initialize population
    population = generate_population(population_size, min_value, max_value)
    
    # Evolution loop
    for generation in range(num_generations):
        # Evaluate population fitness
        fitness = evaluate_population(population)
        
        # Select parents
        parents = tournament_selection(population, fitness, tournament_size)
        
        # Perform crossover
        offspring = crossover(parents)
        
        # Perform mutation
        offspring = mutate(offspring, mutation_rate, min_value, max_value)
        
        # Replace the population with the new generation
        population = offspring
    
    # Select the best individual from the final population
    best_individual = max(zip(population, fitness), key=lambda x: x[1])[0]
    return best_individual

# Example usage
best_solution = genetic_algorithm(population_size=100, num_generations=1000, tournament_size=5, mutation_rate=0.1, min_value=-100, max_value=100)
print("Best solution:", best_solution)
print("Maximum value:", objective_function(best_solution))