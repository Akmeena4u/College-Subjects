
Conversation opened. 2 messages. 2 messages unread.

Skip to content
Using NATIONAL INSTITUTE OF TECHNOLOGY WARANGAL Mail with screen readers

1 of 71
Fwd:
Inbox

Anurag Raj <ar912030@student.nitw.ac.in>
Attachments
7:30 PM (3 hours ago)
to Soumy, B_29, aditya, me


---------- Forwarded message ---------
From: Anurag Raj <ar912030@student.nitw.ac.in>
Date: Mon, Oct 16, 2023, 1:12 PM
Subject:
To: 20 ADITYA SHRIVASTAVA <as812020@student.nitw.ac.in>, A01_Abhishek <aa852040@student.nitw.ac.in>, yatharth garg <yg852022@student.nitw.ac.in>, 07 Anirudh Gupta <ag832024@student.nitw.ac.in>, Anshu Kumar Agrawal <ak822068@student.nitw.ac.in>




6
 Attachments
  •  Scanned by Gmail

aditya
Attachments
9:52 PM (45 minutes ago)
to me

 

 

Sent from Mail for Windows


6
 Attachments
  •  Scanned by Gmail
<-------------------------PERCEPTRON LEARNING--------------------------->

import numpy as np
import pandas as pd

class Perceptron:
  def __init__(self, n_inputs):
    # Initialize Weights, including weights for bias term as well
    self.n = n_inputs+1
    self.weights = (np.random.rand(self.n)-0.5)*2

  def train(self, x,y,  n_iter=1000):
    prev_weights = np.copy(self.weights)
    x = np.array(x)
    # Adding a column of 1s for bias weight
    x = np.c_[np.ones(len(x)) , x]
    for _ in range(n_iter):
      # For each tupple input
      for i in range(len(x)):
        pred_y = np.sum(self.weights*x[i])
        # if xi belongs to P, but we are predicting it as N
        if y[i]==1 and pred_y < 0:
          self.weights += x[i]
        # if xi belongs to N, but we are predicting it as P
        elif y[i]==0 and pred_y>=0:
          self.weights -= x[i]
      # Convergence Condition, i.e. there is no change in weights
      if np.all(self.weights == prev_weights):
        break
      prev_weights = np.copy(self.weights)

  def predict(self, x):
    x = np.c_[np.ones(len(x)) , np.array(x)]
    y = self.weights*x
    ret = []
    for yi in y:
      if np.sum(yi) < 0:
        ret.append(0)
      else:
        ret.append(1)
    return ret

# AND with 2 inputs

p = Perceptron(2)
x = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]
p.train(x,y)
p.predict(x) ,p.weights


# AND with 3 inputs

p1 = Perceptron(3)
x1 = [[i,j,k] for i in range(2) for j in range(2) for k in range(2)]
y1 = [0,0,0,0,0,0,0,1]
p1.train(x1,y1)
p1.predict(x1) ,p1.weights


# AND with 4 inputs

p7 = Perceptron(4)
x7 = [[i,j,k,m] for i in range(2) for j in range(2) for k in range(2) for m in range(2)]
y7 = np.zeros(16, dtype=int)
y7[15] = 1
p7.train(x7,y7)
p7.predict(x7) , p7.weights


import itertools
from sklearn.metrics import accuracy_score
xx = list(itertools.product(range(2), repeat=10))
yy = np.product(xx,axis=1)
pp = Perceptron(10)
pp.train(xx,yy)
# print("Predicted", np.array(pp.predict(xx)), "Actual", yy)
print("Accuracy :",accuracy_score(yy,pp.predict(xx)))
print("Weights :", pp.weights)



<-----------------------MULTILAYER PERCEPTRON------------------------------>

from
keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizersimport Adam, SGD
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
import numpy as np
import pandas as pd
(x_train , y_train) , (x_test, y_test) = mnist.load_data()
n_labels =len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_test


inp_size = x_train.shape[1]**2
x_train = np.reshape(x_train, [-1, inp_size]).astype('float32') /255
x_test = np.reshape(x_test, [-1, inp_size]).astype('float32') /255

model = Sequential([
Flatten(input_shape=(inp_size,)),
Dense(128, activation='relu'),
Dense(n_labels, activation='softmax')
])
model.compile(
optimizer=Adam(),
loss=CategoricalCrossentropy(from_logits=True),
metrics=[CategoricalAccuracy()]
)
model.fit(x_train, y_train ,epochs=10)
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy : ", acc)
print('Accuracy of testing data :', acc)

model2 = Sequential([
Flatten(input_shape=(inp_size,)),
Dense(128, activation='relu'),
Dense(64, activation='relu'),
Dense(n_labels, activation='softmax')
])
model2.compile
(
optimizer=Adam(),
loss=CategoricalCrossentropy(from_logits=True),
metrics=[CategoricalAccuracy()]
)
model2.fit(x_train, y_train ,epochs=10)
loss2 , acc2 = model2.evaluate(x_test, y_test)
print("Accuracy of model with more layers: ", acc2)


model3 = Sequential([
Flatten(input_shape=(inp_size,)),
Dense(512, activation='relu'),
Dense(n_labels, activation='softmax')
])
model3.compile(
optimizer=Adam(),
loss=CategoricalCrossentropy(from_logits=True),
metrics=[CategoricalAccuracy()]
)
model3.fit(x_train, y_train ,epochs=10)
loss3 , acc3 = model3.evaluate(x_test, y_test)
print("Accuracy of model with more layers: ", acc3)

print("Summary of First Model: ")
model.summary()

print("Summary of Second Model: ")
model2.summary()

print("Summary of Third Model: ")
model3.summary()


model4 = Sequential([
Flatten(input_shape=(inp_size,)),
Dense(128, activation='relu'),
Dense(n_labels, activation='softmax')
])
model4.compile
(
optimizer=SGD(),
loss=CategoricalCrossentropy(),
metrics=[CategoricalAccuracy()]
)
model4.fit(x_train, y_train ,epochs=6)
loss4 , acc4 = model4.evaluate(x_test, y_test)
print("Accuracy of model with SGD Optimizer: ", acc4)


<----------------------PCA - EEG and IMAGE------------------------>

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv ('HaLTSubjectA.csv')

data.shape

pca = PCA(n_components=2)
eeg_data_reduced = pca.fit_transform(data)
eeg_data_reduced.shape

plt.plot (data)

plt.plot (eeg_data_reduced)

from skimage import io, color
image_path = "pic.jpg"
image = io.imread(image_path)
gray_image = color.rgb2gray(image)

# Display the original image
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

height, width = gray_image.shape
flattened_image = gray_image.reshape(-1, 1)
pca = PCA(n_components=10, whiten=True)
reduced_image = pca.fit_transform(gray_image)
reconstructed_image = pca.inverse_transform(reduced_image).reshape(height, width)
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Image after PCA')
plt.axis('off')
plt.tight_layout()
plt.show()

height, width

gray_image.shape




<-------------------------AUTOENCODERS------------------------------->

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

def preprocess(array):
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

def noise(array):
    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
    loc=0.0, scale=1.0, size=array.shape
    )
    return np.clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.show()

(train_data, _), (test_data, _) = mnist.load_data()
train_data = preprocess(train_data)
test_data = preprocess(test_data)
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)
display(train_data, noisy_train_data)

input = layers.Input(shape=(28, 28, 1))
# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

autoencoder.fit(
x=train_data,
y=train_data,
epochs=10,
batch_size=128,
shuffle=True,
validation_data=(test_data, test_data),
)

predictions = autoencoder.predict(test_data)
display(test_data, predictions)

autoencoder.fit(
x=noisy_train_data,
y=train_data,
epochs=20,
batch_size=128,
shuffle=True,
validation_data=(noisy_test_data, test_data),
)

predictions = autoencoder.predict(noisy_test_data)
display(noisy_test_data, predictions)


from skimage.metrics import peak_signal_noise_ratio,structural_similarity,mean_squared_error

def calculate_metrics(original_images, denoised_images):
    snr = 20 * np.log10(np.linalg.norm(original_images) / np.linalg.norm(original_images - denoised_images))
    psnr = peak_signal_noise_ratio(original_images, denoised_images, data_range=1.0)
    ssim = structural_similarity(original_images, denoised_images, multichannel=True)
    rmse = np.sqrt(mean_squared_error(original_images, denoised_images))
    return snr, psnr, ssim, rmse

snr, psnr, ssim, rmse = calculate_metrics(noisy_test_data, predictions)

print(f'SNR: {snr}')
print(f'PSNR: {psnr}')
print(f'SSIM: {ssim}')
print(f'RMSE: {rmse}')



<-----------------------------------TSP using GA------------------------------------>

import numpy as np
import random

# Define the TSP problem: a list of cities with their (x, y) coordinates
cities = {
    "A": (0, 0),
    "B": (1, 3),
    "C": (5, 8),
    "D": (10, 6),
    "E": (8, 2)
}

# Genetic Algorithm Parameters
population_size = 50
mutation_rate = 0.01
num_generations = 100

def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Calculate total distance of a tour
def tour_distance(tour):
    total_distance = 0
    for i in range(len(tour) - 1):
        total_distance += distance(tour[i], tour[i + 1])
    total_distance += distance(tour[-1], tour[0])  # Return to the starting city
    return total_distance

def initialize_population(pop_size):
    population = []
    cities_list = list(cities.keys())
    print(cities_list)
    for _ in range(pop_size):
        tour = random.sample(cities_list, len(cities_list))
        population.append(tour)
    return population

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + [city for city in parent2 if city not in parent1[:point]]
    child2 = parent2[:point] + [city for city in parent1 if city not in parent2[:point]]
    return child1, child2

# Mutation
def mutate(tour):
    idx1, idx2 = random.sample(range(len(tour)), 2)
    tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# Genetic Algorithm
def genetic_algorithm(pop_size, mutation_rate, generations):
    population = initialize_population(pop_size)
    #print(population)
    for generation in range(generations):
        new_population = []
        #print(new_population)
        # Selection and crossover
        for _ in range(pop_size // 2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([child1, child2])
        
        # Mutation
        for i in range(len(new_population)):
            if random.random() < mutation_rate:
                new_population[i] = mutate(new_population[i])
        
        population = new_population
    
    best_tour = min(population, key=tour_distance)
    best_distance = tour_distance(best_tour)
    
    return best_tour, best_distance

best_tour, best_distance = genetic_algorithm(population_size, mutation_rate, num_generations)

print("Best tour:", best_tour)
print("Best distance:", best_distance)



<-------------------------------TSP using ACO----------------------------->

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



<-------------------------------CLUSTERING using PSO-------------------------->

import random

# Define the objective function (quadratic)
def objective_function(x):
    return x**2

# PSO parameters
num_particles = 20
num_dimensions = 1
max_iterations = 50
c1 = 2.0  # Cognitive parameter
c2 = 2.0  # Social parameter
w = 0.7   # Inertia weight

# Initialize particles and velocities randomly
particles = []
velocities = []
for _ in range(num_particles):
    particle = [random.uniform(-10, 10) for _ in range(num_dimensions)]
    particles.append(particle)
    velocities.append([0.0] * num_dimensions)

# Initialize global best position and value
global_best_position = particles[0]
global_best_value = objective_function(particles[0][0])

# PSO main loop
for iteration in range(max_iterations):
    for i in range(num_particles):
        particle = particles[i]
        current_value = objective_function(particle[0])
        
        if current_value < global_best_value:
            global_best_value = current_value
            global_best_position = particle
        
        # Update velocity and position
        for j in range(num_dimensions):
            r1 = random.random()
            r2 = random.random()
            cognitive_component = c1 * r1 * (global_best_position[j] - particle[j])
            social_component = c2 * r2 * (global_best_position[j] - particle[j])
            velocities[i][j] = w * velocities[i][j] + cognitive_component + social_component
            particle[j] += velocities[i][j]
            
# Print the best solution found
print("Best solution found:", global_best_position)
print("Best objective value:", global_best_value)



<----------------------------IMAGE RECONCSTR. using GA---------------------------------->

from deap import base, creator, tools, algorithms
from random import randint, random, gauss
from PIL import Image, ImageDraw
from functools import partial
from math import sqrt
import numpy


PIC = Image.open('ga.jpg')
SIZE = 100
NUMBER_OF_TRIANGLES = 50
POPULATION = 40
NGEN = 1000
POLY = 3


def gen_one_triangle():
    return (tuple([(randint(0, SIZE), randint(0, SIZE)) for i in range(POLY)]),
            randint(0,255), randint(0,255), randint(0,255), randint(0,30))
            #0, 0, 0, 0)


def triangles_to_image(triangles):
    im = Image.new('RGB', (SIZE, SIZE), (255, 255, 255))
    for tri in triangles:
        mask = Image.new('RGBA', (SIZE, SIZE))
        draw = ImageDraw.Draw(mask)
        draw.polygon(tri[0], fill=tri[1:])
        im.paste(mask, mask=mask)
        del mask, draw
    return im


def evaluate(im1, t2):
    im2 = triangles_to_image(t2)
    pix1, pix2 = im1.load(), im2.load()
    ans = 0
    for i in range(SIZE):
        for j in range(SIZE):
            a1, a2, a3 = pix1[i, j]
            b1, b2, b3 = pix2[i, j]
            ans += (a1 - b1) ** 2 + (a2 - b2) ** 2 + (a3 - b3) ** 2
    return 1 - (1. * sqrt(ans) / sqrt(SIZE * SIZE * 3 * 255 * 255)),


def mutate(triangles):
    e0 = evaluate(PIC, triangles)
    for i in range(10):
        tid = randint(0, NUMBER_OF_TRIANGLES - 1)
        oldt = triangles[tid]

        t = list(oldt)
        p = randint(0, 2 * POLY + 4 - 1)
        if p < 2 * POLY:
            points = list(t[0])
            pnt = list(points[p // 2])
            #pnt[p%2] = max(0, min(SIZE, gauss(pnt[p%2], 10)))
            pnt[p % 2] = randint(0, SIZE)
            points[p // 2] = tuple(pnt)
            t[0] = tuple(points)
        else:
            p -= 2 * POLY - 1
            #t[p] = max(0, min(255, int(gauss(t[p], 20))))
            t[p] = randint(0, 255)

        triangles[tid] = tuple(t)
        if evaluate(PIC, triangles) > e0:
            break
        else:
            triangles[tid] = oldt
    return triangles,

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr", gen_one_triangle)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr, NUMBER_OF_TRIANGLES)
toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual)

toolbox.register("evaluate", partial(evaluate, PIC))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    pop = toolbox.population(n=POPULATION)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", numpy.std)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)
    stats.register("min", numpy.min)

    try:
        pop, log = algorithms.eaSimple(
            pop, toolbox, cxpb=0.5, mutpb=0.1, ngen=NGEN, stats=stats,
            halloffame=hof, verbose=True)
    finally:
        open('result.txt', 'w').write(repr(hof[0]))
        triangles_to_image(hof[0]).save('result1.bmp')

if __name__ == '__main__':
    main()


<-------------------------------TSP - MOO-------------------------------------->

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



<------------ACO - feature reduction followed by reconstrt of image----------------->

import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Step 1: Preprocessing
# A. Load the image
image = io.imread('pic.jpg')

# B. Convert to grayscale
gray_image = color.rgb2gray(image)

# C. Flatten the image into a 1D vector
flattened_image = gray_image.flatten()

# Step 2: Feature Reduction with ACO
# A. Define the problem of feature reduction
# Let's assume you want to select N pixels from the flattened image.
num_features_to_select = 1000  # Adjust as needed

# B. Initialize ACO parameters
num_ants = 10
num_iterations = 50
pheromone_levels = np.ones(len(flattened_image))
pheromone_decay = 0.5
alpha = 1.0  # Controls the importance of pheromone levels
beta = 1.0   # Controls the importance of heuristic information

# Initialize a list to store the selected feature indices
selected_indices = []

# C. Implement the ACO algorithm
for iteration in range(num_iterations):
    for ant in range(num_ants):
        # Implement ant's feature selection logic here
        # For simplicity, we randomly select features (pixels) in this example
        selected_features = np.random.choice(len(flattened_image), num_features_to_select, replace=False)

        # Calculate the quality of the selected subset (you should replace this with your evaluation)
        subset_quality = np.mean(flattened_image[selected_features])

        # Update pheromone levels based on the quality of the selected subset
        pheromone_levels[selected_features] += alpha * subset_quality

    # Apply pheromone decay
    pheromone_levels *= pheromone_decay

    # Select the best feature subset from this iteration
    best_subset = np.argsort(pheromone_levels)[-num_features_to_select:]
    selected_indices = best_subset

# Step 3: Reconstructing the Image
# B. Create a new image with only the selected pixels
reconstructed_image = np.zeros_like(gray_image)
for index in selected_indices:
    row, col = np.unravel_index(index, gray_image.shape)
    reconstructed_image[row, col] = gray_image[row, col]

# Step 4: Display the Reconstructed Image
# A. Display original and reconstructed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.show()


from skimage.metrics import mean_squared_error, structural_similarity

# Load the original and reconstructed images
original_image = gray_image  # Assuming gray_image is the original image
# reconstructed_image is the image you obtained from the ACO-based feature reduction

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(original_image, reconstructed_image)

# Compute Structural Similarity Index (SSIM)
ssim = structural_similarity(original_image, reconstructed_image)

# Print the quality metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Structural Similarity Index (SSIM): {ssim:.2f}")



<--------------------------Decision Tree - Yale Faces dataset------------------------->

import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import pandas as pd
import glob

dir_path = "C:\\Users\\shubh\\Downloads\\archive\\data"
class_labels=os.listdir(dir_path)
print(class_labels)

num_classes=len(class_labels)
print(num_classes)

list_files = glob.glob("C:\\Users\\shubh\\Downloads\\archive\\data//*");
print(list_files[4])

new = []
for i in range(0,len(list_files)):
    new.append(list_files[i])
print(len(new))

## For getting the standard size of each image, assuming that all the images are of the same size
image = Image.open(new[32])
standard_size = image.size
display(image)

image = Image.open(new[25])
standard_size = image.size
display(image)

new.pop(0)
list_files.pop(0)
labels = []
D = []
for i in range(len(new)):
    image = Image.open(new[i])
    ## Gray Scale Conversion
    image = image.convert('L')
    im = np.array(image)
    # Row-Major Format
    im = im.ravel(order='K')
    D.append(im)
D = np.array(D)

print(D)
print(len(D),len(D[0]))
print(D[20])

print(len(D),len(D[0]))

# D= np.delete(D,0,0)
class_labels.pop(0)
len(class_labels)

labels = []
for i in range(len(class_labels)):
    labels.append(class_labels[i].split(".")[0])
print(labels)

D = np.array(D)
print(len(D),len(D[0]))

from sklearn.model_selection import train_test_split
train_D, test_D, train_y, test_y = train_test_split(D, labels,test_size=0.2, random_state=0)
# train_P,test_P,train_y_p,test_y_p = train_test_split(P,labels,test_size=0.2,random_state=0)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_D, train_y)
y_pred = clf.predict(test_D)

y_pred

test_y

from sklearn.metrics import accuracy_score

error_rate = 1 - accuracy_score(y_pred,test_y)
print('Error rate',error_rate)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
plot_tree(clf) 
plt.show() 


average_face_ini = np.mean(D,axis=0)
average_face_ini.shape
average_face_ini

standard_size

average_face = average_face_ini.reshape(243,320)
im =  Image.fromarray(average_face)
import matplotlib.pyplot as plt
plt.gray()
plt.imshow(im)
plt.show()

fin = D - average_face_ini
a = np.mean(fin,axis=1)
print(np.sum(fin))

fin

from sklearn.decomposition import PCA
pca = PCA()
principalComponents = pca.fit_transform(fin)

len(principalComponents)

x = [1,2,3,4,6]
y = pca.explained_variance_ratio_[:5]
plt.plot(x,y)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()

E= pca.components_[:10]
E.shape

for i in range(len(E)):
    r = E[i].reshape(243,320)
    im =  Image.fromarray(r)
   
    plt.gray()
    plt.imshow(np.asarray(im))
    plt.show()

E = np.asmatrix(E)
D= np.asmatrix(D)
P = np.dot(D,E.T)
P.shape
P=np.array(P)

train_P, test_P, train_y, test_y = train_test_split(P, labels,test_size=0.2, random_state=0)
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(train_P, train_y)
y_pred2 = clf2.predict(test_P)
print(y_pred2)
print(test_y)

error_rate_pca = 1 - accuracy_score(y_pred2,test_y)
print('Error rate with pca',error_rate_pca)

plt.figure(figsize=(20, 20))
plot_tree(clf2) 
plt.show() 



<------------------------------Zero Assgn---------------------------------->


from sklearn.datasets import load_digits
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset=load_digits(

X=dataset.data
Y=dataset.target

print(X)
print(Y)

samples=X.shape[0]
features=X.shape[1]
print("Number of samples in given dataset : ", samples)
print("Number of features in given datset : ", features)

trainImg, testImg, trainClass, testClass = train_test_split(X, Y, test_size=0.2)

pd.DataFrame(X)

clf=tree.DecisionTreeClassifier(max_depth=30)
clf=clf.fit(trainImg, trainClass)

Y_pred= clf.predict(testImg)

matches=0
for i in range(len(testClass)):
	if(testClass[i]==Y_pred[i]):
		matches+=1
print("Number of accurate matches is : ", matches, ", out of ", len(testClass))

def plot_images(images, titles, h, w, rows=3, cols=4):
	plt.figure(figsize=(2*cols, 3*rows))
	plt.subplots_adjust(bottom =0, left=0.01, right=0.99, top=0.90, hspace=0.35)
	for i in range(rows*cols):
		plt.subplot(rows, cols, i+1)
		plt.imshow(images[i].reshape(h,w), cmap=plt.cm.gray)
		plt.title(titles[i], size12)
		plt.xticks(())
		plt.yticks(())

def title(y_pred, y_test, i):
	pred_name=y_pred[i]
	true_name=y_test[i]
	return 

<------------------------CNN - Conv2D NN ------------------------------->

import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Assuming 10 classes for classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# You can print a summary of the model architecture
model.summary()
LM - 345622A10.txt
Displaying LM - 345622A10.txt. 
