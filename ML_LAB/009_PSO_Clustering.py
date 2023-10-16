'''3.How does the Particle Swarm Optimization (PSO) algorithm offer a solution to the clustering problem? Could you elucidate the core mechanisms of PSO that
contribute to the discovery of optimal or near-optimal cluster configurations in various clustering scenarios.

'''

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
