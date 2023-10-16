'''2.How is the Traveling Salesman Problem (TSP) tackled by the Ant Colony Optimization (ACO) algorithm? Could you elaborate on the particular ACO
mechanisms that contribute to producing efficient solutions for optimizing routes in the context of TSP.

it is an implementation of the Ant Colony Optimization (ACO) algorithm to solve the Traveling Salesman Problem (TSP) for a small set of 5 cities. It does the following:

Creates a graph representing cities and their distances.
Sets ACO parameters (number of ants, iterations, evaporation rate, etc.).
Initializes pheromone levels on edges.
Runs ACO for the specified iterations.
Updates pheromone levels based on path quality.
Prints the best path and distance in each iteration.
The goal is to find the shortest path that visits each city once and returns to the starting city. Pheromone levels guide the ants toward shorter paths, and the algorithm aims to converge on the optimal solution over multiple iterations.


'''

import networkx as nx
import random
import math
import matplotlib.pyplot as plt

import networkx as nx
import random
import matplotlib.pyplot as plt

# Create a graph using NetworkX
G = nx.Graph()

num_cities = 5

# Generate random city positions
cities = {chr(65 + i): (random.uniform(0, 10), random.uniform(0, 10)) for i in range(num_cities)}

# Add cities as nodes to the graph
for city, pos in cities.items():
    G.add_node(city, pos=pos)


 
# Calculate distances between cities and add edges to the graph
for city1, pos1 in cities.items():
    for city2, pos2 in cities.items():
        if city1 != city2:
            distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            G.add_edge(city1, city2, weight=distance)

# ACO parameters
num_ants = 10
num_iterations = 50
evaporation_rate = 0.5
Q = 100
alpha = 1
beta = 3

# Initialize pheromone levels on edges
pheromones = {(city1, city2): 1 for city1 in cities for city2 in cities if city1 != city2}

# ACO algorithm
for iteration in range(num_iterations):
    paths = []

    # Generate paths for each ant
    for ant in range(num_ants):
        current_city = random.choice(list(cities.keys()))
        path = [current_city]
        visited_cities = set([current_city])

        # Construct a path for the ant
        for _ in range(len(cities) - 1):
            neighbor_cities = [city for city in G.neighbors(current_city)]
            feasible_cities = [city for city in neighbor_cities if city not in visited_cities]
            if not feasible_cities:
                break  # All cities visited
            next_city = max(feasible_cities, key=lambda city: pheromones[(current_city, city)]**alpha * (1 / G[current_city][city]['weight'])**beta)
            path.append(next_city)
            visited_cities.add(next_city)
            current_city = next_city

        paths.append((path, sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1)) + G[path[-1]][path[0]]['weight']))

    # Update pheromone levels
    for edge in G.edges:
        pheromones[edge] *= (1 - evaporation_rate)

    for path, distance in paths:
        for i in range(len(path) - 1):
            pheromones[(path[i], path[i+1])] += Q / distance
        pheromones[(path[-1], path[0])] += Q / distance

    # Find the best path in this iteration
    best_path, best_distance = min(paths, key=lambda x: x[1])

    print(f"Iteration {iteration+1}: Best Path = {best_path}, Distance = {best_distance}")
