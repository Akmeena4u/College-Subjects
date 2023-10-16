'''1.How does the utilization of genetic algorithm contribute to the iterative reconstruction of images, and what parameters or factors play a crucial role in
achieving accurate and efficient image reconstruction outcomes.'''




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




'''This code uses a genetic algorithm to evolve a set of triangles to approximate an input image. Here's a concise overview:

Initialization and Constants:

Import libraries, set constants like image (PIC), image size (SIZE), number of triangles (NUMBER_OF_TRIANGLES), population size (POPULATION), generations (NGEN), and polygon vertices (POLY).
Triangle Generation and Image Transformation:

gen_one_triangle() creates random triangles.
triangles_to_image() converts triangles into an image.
Fitness Function:

evaluate() measures the difference between the input image and the image created by triangles.
Mutation Operator:

mutate() adjusts triangle properties to improve fitness.
DEAP Setup:

Define the fitness and individual types.
Configure DEAP toolbox for initialization, evaluation, crossover, mutation, and selection.
Main Evolutionary Algorithm:

Initialize a population and setup statistics and a hall of fame.
Run the evolutionary algorithm (eaSimple) with specified parameters.
Result and Execution Block:

Save the best individual's data and image representation.
In summary, this code evolves triangles to approximate an image using a genetic algorithm. The best-fit solution is saved and displayed as an output image.'''
