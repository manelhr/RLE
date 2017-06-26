from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import pandas as pd
import numpy as np


def nsga2_robust_explanation(explanation):

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))     # creates fitness function
    creator.create("Individual", list, fitness=creator.FitnessMulti)      # register type of individual

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=10)
    print(pop)

nsga2_robust_explanation()
