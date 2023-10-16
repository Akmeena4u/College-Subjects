'''Consider a scenario where a salesperson needs to visit multiple cities to complete their sales route efficiently. However, there are two conflicting
objectives to consider minimizing the total distance traveled and minimizing the total time taken. Implement this problem as a Multi-Objective Traveling
Salesman Problem (TSP-MOO).

it is a Python implementation of a multi-objective genetic algorithm to solve a Traveling Salesman Problem (TSP) with two objectives: minimizing both distance and cost.
The code uses the DEAP (Distributed Evolutionary Algorithms in Python) library for implementing the genetic algorithm. '''

import random
import numpy as np
from deap import base, creator, tools, algorithms

# Define the problem as a multi-objective problem (minimize both distance and cost with equal weightage)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Define the function to evaluate an individual's fitness (distance and cost)
def evaluate(individual, distance_matrix, cost_matrix):
    distance = 0
    cost = 0
    num_cities = len(individual)
    
    for i in range(num_cities):
        from_city = individual[i]
        to_city = individual[(i + 1) % num_cities]
        distance += distance_matrix[from_city][to_city]
        cost += cost_matrix[from_city][to_city]
    
    return distance, cost

# Create random distance and cost matrices (replace these with your own data)
num_cities = 10
distance_matrix = np.random.rand(num_cities, num_cities)
cost_matrix = np.random.rand(num_cities, num_cities)

# Define the genetic algorithm parameters
population_size = 100
generations = 100
cx_prob = 0.7  # Crossover probability
mut_prob = 0.2  # Mutation probability

# Create the toolbox
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(1, num_cities), num_cities - 1)  # Exclude city 0 initially
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)  # Register the evaluation function



# Main evolutionary algorithm loop
if __name__ == "__main__":
    best_path = None
    best_distance = float("inf")
    best_cost = float("inf")
    population = toolbox.population(n=population_size)
    
    # Ensure that each individual starts from city 0
    for ind in population:
        ind.insert(0, 0)
    
    for gen in range(generations):
        # Evaluate the population
        fitness_values = [toolbox.evaluate(ind, distance_matrix, cost_matrix) for ind in population]
        for ind, fit in zip(population, fitness_values):
            ind.fitness.values = fit
            if fit[0] < best_distance and fit[1] < best_cost:
                best_distance = fit[0]
                best_cost = fit[1]
                best_path = ind[:]
        
        # Print distance and cost at each iteration
        print(f"Generation {gen + 1}: Distance={best_distance}, Cost={best_cost}")
        
        # Create offspring using genetic operators
        offspring = algorithms.varAnd(population, toolbox, cxpb=cx_prob, mutpb=mut_prob)
        fits = [toolbox.evaluate(ind, distance_matrix, cost_matrix) for ind in offspring]
        
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
            if fit[0] < best_distance and fit[1] < best_cost:
                best_distance = fit[0]
                best_cost = fit[1]
                best_path = ind[:]
        
        # Select the next generation
        population = toolbox.select(offspring + population, k=population_size)
        
    # Ensure that the best path ends at city 0
    best_path.append(0)
    
    # Get the Pareto front of solutions (trade-off between distance and cost)
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    # Print the best path and other information
    print("Best Distance:", best_distance)
    print("Best Cost:", best_cost)
    print("Best Path:", best_path)
