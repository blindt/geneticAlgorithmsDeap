# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import multiprocessing
import random

from deap import base
from deap import creator
from deap import tools

toolbox = base.Toolbox()


def set_type_of_find_element(findtype):
    if findtype == 1:
        creator.create("Fitness", base.Fitness, weights=(1.0,))
    elif findtype == 2:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))


def set_fitness_function():
    toolbox.register("evaluate", fitnessFunction)


def set_selection_method(seltype, parameter):
    if seltype == 1:
        toolbox.register("select", tools.selTournament, tournsize=parameter)
    elif seltype == 2:
        toolbox.register("select", tools.selRandom)
    elif seltype == 3:
        toolbox.register("select", tools.selBest)
    elif seltype == 4:
        toolbox.register("select", tools.selWorst)
    elif seltype == 4:
        toolbox.register("select", tools.selRoulette)


def set_crossing_method(crosstype):
    if crosstype == 1:
        toolbox.register("mate", tools.cxOnePoint)
    if crosstype == 2:
        toolbox.register("mate", tools.cxUniform)


def set_mutation_method(muttype):
    if muttype == 1:
        toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=0.5)
    if muttype == 2:
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
    if muttype == 3:
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.5)


def set_map_for_toolbox(pool):
    toolbox.register("map", pool.map)


def fitnessFunction(individual):
    result = (individual[0] + 2 * individual[1] - 7) ** 2 + (2 * individual[0] + individual[1] - 5) ** 2
    return result,


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))

    return icls(genome)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


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
    cross_type = int(input())

    print("Select type of mutation: ")
    print("1. Gaussian")
    print("2. Shuffle indexes")
    print("2. Flip bit")
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


size_population = 100
probability_mutation = 0.2
probability_crossover = 0.8
number_iteration = 100


def run_algorithm(pop):
    g = 0
    number_elitism = 1
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
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        #
    print("-- End of (successful) evolution --")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    select, find_element, cross, mut = main_ui()
    pool = multiprocessing.Pool(processes=4)
    pop = set_algorithm(select, find_element, cross, mut)
    set_map_for_toolbox(pool)
    run_algorithm(pop)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
