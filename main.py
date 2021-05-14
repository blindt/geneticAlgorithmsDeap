# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import multiprocessing
import random
import time

from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

toolbox = base.Toolbox()


def arithmetic_crossing(ind1, ind2):
    k = random.uniform(0, 1)

    new_ind1 = tuple(k * x for x in ind1) + tuple((1 - k) * x for x in ind2)
    new_ind2 = tuple((1 - k) * x for x in ind1) + tuple(k * x for x in ind2)
    return new_ind1, new_ind2


def heuristic_crossing(ind1, ind2):
    k = random.uniform(0, 1)
    if ind2.fitness.values >= ind1.fitness.values:
        x1new = k * (ind2[0] - ind1[0]) + ind2[0]
        y1new = k * (ind2[1] - ind1[1]) + ind2[1]
        new_ind = (x1new, y1new)
    else:
        x1new = k * (ind1[0] - ind2[0]) + ind1[0]
        y1new = k * (ind1[1] - ind2[1]) + ind1[1]
        new_ind = (x1new, y1new)
    return new_ind


def set_type_of_find_element(findtype):
    if findtype == 1:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
    elif findtype == 2:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))


def set_fitness_function():
    toolbox.register("evaluate", fitness_function)


def set_selection_method(seltype, parameter):
    if seltype == 1:
        toolbox.register("select", tools.selTournament, tournsize=parameter)
    elif seltype == 2:
        toolbox.register("select", tools.selRandom)
    elif seltype == 3:
        toolbox.register("select", tools.selBest)
    elif seltype == 4:
        toolbox.register("select", tools.selWorst)
    elif seltype == 5:
        toolbox.register("select", tools.selRoulette)


def set_crossing_method(crosstype):
    if crosstype == 1:
        toolbox.register("mate", tools.cxOnePoint)
    if crosstype == 2:
        toolbox.register("mate", tools.cxUniform, indpb=0.5)
    if crosstype == 3:
        toolbox.register("mate", arithmetic_crossing)
    if crosstype == 4:
        toolbox.register("mate", heuristic_crossing)


def set_mutation_method(muttype):
    if muttype == 1:
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.5)
    if muttype == 2:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    if muttype == 3:
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)


def set_map_for_toolbox(pool):
    toolbox.register("map", pool.map)


def fitness_function(individual):
    result = (1.5 - individual[0] + individual[0]*individual[1])**2\
             + (2.25 - individual[0] + individual[0] * individual[1]**2)**2\
             + (2.625 - individual[0] + individual[0]*individual[1]**3)**2
    #result = (individual[0] + 2 * individual[1] - 7) ** 2 + (2 * individual[0] + individual[1] - 5) ** 2
    return result,


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))

    return icls(genome)


def main_ui():
    print("Select type of algorithm: ")
    print("1. Maximization")
    print("2. Minimization")
    find_element_type = int(input())

    print("Select type of selection: ")
    print("1. Tournament")
    print("2. Random")
    print("3. Best")
    print("4. Worst")
    print("5. Roulette")
    select_type = int(input())

    print("Select type of crossing: ")
    print("1. One Point")
    print("2. Uniform")
    print("3. Arithmetic")
    print("4. Heuristic")
    cross_type = int(input())

    print("Select type of mutation: ")
    print("1. Gaussian")
    print("2. Shuffle indexes")
    print("3. Flip bit")
    mut_type = int(input())

    return select_type, find_element_type, cross_type, mut_type


creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)
toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def set_algorithm(select_type, find_element_type, cross_type, mut_type):
    #set_type_of_find_element(find_element_type)
    set_fitness_function()
    set_selection_method(select_type, 4)
    set_crossing_method(cross_type)
    set_mutation_method(mut_type)
    pop = toolbox.population(n=size_population)
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return pop


size_population = 1000
probability_mutation = 0.2
probability_crossover = 0.8
number_iteration = 1000


def generate_plots(best, mean, std):
    plt.xlabel("epoch")
    plt.ylabel("function value")
    plt.title("best function values")
    plt.plot(best)
    plt.show()

    plt.xlabel("epoch")
    plt.ylabel("mean")
    plt.title("mean values")
    plt.plot(mean)
    plt.show()

    plt.xlabel("epoch")
    plt.ylabel("std value")
    plt.title("std values")
    plt.plot(std)
    plt.show()


def run_algorithm(pop):
    g = 0
    number_elitism = 1
    best_list = []
    mean_list = []
    std_list = []
    start_time = time.time()
    while g < number_iteration:
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        list_elitism = []
        for x in range(0, number_elitism):
            list_elitism.append(tools.selBest(pop, 1)[0])

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < probability_crossover:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < probability_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring + list_elitism

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]
        mean_list.append(mean)
        std_list.append(std)
        best_list.append(best_ind.fitness.values)
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        #
    print("-- End of (successful) evolution --")
    print("time is: ", time.time() - start_time)
    generate_plots(best_list, mean_list, std_list)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    select, find_element, cross, mut = main_ui()

    pool = multiprocessing.Pool(processes=4)
    pop = set_algorithm(select, find_element, cross, mut)
    set_map_for_toolbox(pool)
    run_algorithm(pop)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
